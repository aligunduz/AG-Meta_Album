"""Small CPU algorithm and submission integration checks; no dataset needed."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import helpers_fo_proto_maml as helpers
import model as baseline


class Tiny(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.model = nn.ModuleDict({"out": nn.Linear(4, 5)})

    def forward_weights(self, x, weights, embedding=False):
        z = F.linear(x, weights[0], weights[1]).tanh()
        return z if embedding else F.linear(z, weights[-2], weights[-1])


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(12)
        self.model = Tiny()
        self.model.model["out"] = nn.Identity()
        self.weights = list(self.model.parameters())
        self.cfg = baseline.read_config()["method_config"]
        self.x = torch.randn(6, 3)
        self.y = torch.tensor([1, 0, 1, 1, 0, 1])

    def test_probabilities_unequal_shots_and_order(self):
        features = self.model.forward_weights(self.x, self.weights, True)
        w, b = helpers.prototype_head(features, self.y)
        centers = torch.stack([features[self.y == k].mean(0) for k in range(2)])
        q = torch.randn(5, 4)
        torch.testing.assert_close(w, 2 * centers)
        torch.testing.assert_close(b, -centers.square().sum(1))
        torch.testing.assert_close(F.linear(q, w, b).softmax(1),
                                   (-torch.cdist(q, centers).square()).softmax(1))

    def test_prototype_gradient_without_query_encoder_path(self):
        features = self.model.forward_weights(self.x, self.weights, True)
        features.retain_grad()
        w, b = helpers.prototype_head(features, self.y)
        F.cross_entropy(F.linear(torch.randn(3, 4), w, b),
                        torch.tensor([0, 1, 0])).backward()
        self.assertGreater(features.grad.norm().item(), 0)
        self.assertGreater(self.weights[0].grad.norm().item(), 0)

    def test_first_order_and_preserved_head_jacobian(self):
        original_grad = torch.autograd.grad
        recorded = []
        def observe(*args, **kwargs):
            grads = original_grad(*args, **kwargs)
            self.assertFalse(kwargs["create_graph"])
            self.assertTrue(all(not g.requires_grad and g.grad_fn is None for g in grads))
            recorded.append(grads)
            return grads
        with patch.object(torch.autograd, "grad", side_effect=observe):
            fast = helpers.adapt(self.model, self.weights, self.x, self.y, self.cfg)
        self.assertEqual(len(recorded), 5)
        # Linear objective in W,b isolates their initialization Jacobian even
        # after nonzero updates; a detached head cannot pass this test.
        coeff_w, coeff_b = torch.randn_like(fast[-2]), torch.randn_like(fast[-1])
        actual = original_grad((fast[-2]*coeff_w).sum() + (fast[-1]*coeff_b).sum(), self.weights)
        w, b = helpers.prototype_head(self.model.forward_weights(self.x, self.weights, True), self.y)
        expected = original_grad((w*coeff_w).sum() + (b*coeff_b).sum(), self.weights)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)
        self.assertGreater(actual[0].norm().item(), 0)

    def test_inner_encoder_gradient_holds_head_fixed(self):
        cfg = dict(self.cfg, inner_steps=1, grad_clip=None)
        z = self.model.forward_weights(self.x, self.weights, True)
        w, b = helpers.prototype_head(z, self.y)
        expected = torch.autograd.grad(F.cross_entropy(F.linear(z, w.detach(), b.detach()), self.y), self.weights)
        fast = helpers.adapt(self.model, self.weights, self.x, self.y, cfg)
        for initial, updated, grad in zip(self.weights, fast, expected):
            torch.testing.assert_close(updated, initial - cfg["encoder_lr"] * grad)

    def test_no_query_leakage_and_episode_independence(self):
        query = torch.randn(3, 3, requires_grad=True)
        snapshot = [p.detach().clone() for p in self.weights]
        first = helpers.adapt(self.model, self.weights, self.x, self.y, self.cfg)
        self.assertIsNone(torch.autograd.grad(first[-2].sum(), query, allow_unused=True)[0])
        helpers.adapt(self.model, self.weights, self.x + 2, self.y, self.cfg)
        again = helpers.adapt(self.model, self.weights, self.x, self.y, self.cfg)
        for a, b in zip(first, again):
            torch.testing.assert_close(a, b)
        for a, b in zip(snapshot, self.weights):
            torch.testing.assert_close(a, b)

    def test_variable_way_no_grad_and_labels(self):
        for ways in (2, 7):
            labels = torch.arange(ways).repeat(2).flip(0)
            x = torch.randn(len(labels), 3)
            with torch.no_grad():
                fast = helpers.adapt(self.model, self.weights, x, labels, self.cfg, ways)
                out, _ = helpers.query_loss(self.model, fast, x, labels)
            self.assertEqual(out.shape, (len(labels), ways))
        with self.assertRaises(ValueError):
            helpers.prototype_head(torch.randn(2, 4), torch.tensor([0, 2]))
        with self.assertRaises(ValueError):
            helpers.query_loss(self.model, fast, x, labels + 7)

    def test_config_matches_fomaml(self):
        reference = json.loads((ROOT.parent / "maml/config.json").read_text())
        current = baseline.read_config()
        for key, value in reference.items():
            self.assertEqual(current[key], value)

    def test_train_validation_checkpoint_and_predict(self):
        cfg = baseline.read_config()
        cfg["experiment_config"] = dict(train_iterations=2, validation_tasks=1, validate_every=2)
        task = SimpleNamespace(num_ways=2, support_set=(self.x, self.y, self.y),
                               query_set=(self.x + .2, self.y, self.y))
        generator = lambda count: iter([task] * count)
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=cfg):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))
            meta.log = lambda *a, **kw: None
            before = [p.detach().clone() for p in meta.weights]
            learner = meta.meta_fit(generator, generator)
            self.assertFalse(all(torch.equal(a, b) for a, b in zip(before, meta.weights)))
            self.assertFalse(any("out" in name for name, _ in meta.meta_learner.named_parameters()))
            support = (self.x, self.y, self.y, 2, 3)
            expected = learner.fit(support).predict(self.x)
            with tempfile.TemporaryDirectory() as directory:
                learner.save(directory)
                restored = baseline.MyLearner()
                restored.load(directory)
                torch.testing.assert_close(torch.tensor(expected), torch.tensor(restored.fit(support).predict(self.x)))
                payload = torch.load(Path(directory)/"max-va.pth", weights_only=True)
                self.assertEqual(payload["config"], cfg)
                self.assertEqual(payload["method"], "fo-proto-maml")
            best = {k: v.clone() for k, v in meta.best_state.items()}
            meta.best_score = 2  # next validation cannot improve
            with torch.no_grad():
                meta.weights[0].add_(1)
            meta.meta_valid(generator)
            for k in best:
                torch.testing.assert_close(best[k], meta.best_state[k])

    def test_real_backbone_smoke(self):
        model = baseline.make_encoder(dict(num_classes=2, dev=torch.device("cpu"), num_blocks=18, pretrained=False))
        x = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([1, 0, 0, 1])
        fast = helpers.adapt(model, list(model.parameters()), x, labels, dict(self.cfg, inner_steps=1))
        _, loss = helpers.query_loss(model, fast, x + .1, labels)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_existing_protonet_equivalence(self):
        spec = importlib.util.spec_from_file_location("original_protonet", ROOT.parent / "protonet/helpers_protonet.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        prototypes = original.process_support_set(self.model, self.weights, self.x, self.y, 2)
        old = original.process_query_set(self.model, self.weights, self.x + .3, prototypes)
        fast = helpers.adapt(self.model, self.weights, self.x, self.y, dict(self.cfg, inner_steps=0))
        new = self.model.forward_weights(self.x + .3, fast)
        torch.testing.assert_close(old.softmax(1), new.softmax(1))


if __name__ == "__main__":
    unittest.main()
