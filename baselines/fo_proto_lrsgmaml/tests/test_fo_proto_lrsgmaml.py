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
import helpers_fo_proto_lrsgmaml as helpers
import model as baseline
from low_rank_transport import LowRankTransport
from metrics import log_metrics


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
            optimizer_ids = {id(p) for group in meta.optimizer.param_groups for p in group["params"]}
            self.assertTrue(all(id(p) in optimizer_ids for p in meta.transport.parameters()))
            self.assertTrue({id(p) for p in meta.weights}.isdisjoint(
                {id(p) for p in meta.transport.parameters()}))
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
                for key, value in learner.transport.state_dict().items():
                    torch.testing.assert_close(value, restored.transport.state_dict()[key], rtol=0, atol=0)
                torch.testing.assert_close(torch.tensor(expected), torch.tensor(restored.fit(support).predict(self.x)))
                with torch.no_grad():
                    for p in restored.transport.logits.values():
                        p.fill_(-10)
                    for p in restored.transport.v.values():
                        p.zero_()
                self.assertFalse(torch.equal(torch.tensor(expected),
                                             torch.tensor(restored.fit(support).predict(self.x))))
                payload = torch.load(Path(directory)/"max-va.pth", weights_only=True)
                self.assertEqual(payload["config"], cfg)
                self.assertEqual(payload["method"], "fo-proto-lrsgmaml")
            best = {k: v.clone() for k, v in meta.best_state["lrsg"].items()}
            meta.best_score = 2  # next validation cannot improve
            with torch.no_grad():
                meta.weights[0].add_(1)
            meta.meta_valid(generator)
            for k in best:
                torch.testing.assert_close(best[k], meta.best_state["lrsg"][k])

    def test_real_backbone_smoke(self):
        model = baseline.make_encoder(dict(num_classes=2, dev=torch.device("cpu"), num_blocks=18, pretrained=False))
        x = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([1, 0, 0, 1])
        transport = LowRankTransport(model, baseline.read_config()["lrsg"])
        fast = helpers.adapt(model, list(model.parameters()), x, labels,
                             dict(self.cfg, inner_steps=5), transport=transport)
        _, loss = helpers.query_loss(model, fast, x + .1, labels)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in transport.parameters()))

    def test_lrsg_config_parity(self):
        reference = json.loads((ROOT.parent / "fo_proto_maml/config.json").read_text())
        current = baseline.read_config()
        self.assertEqual(current.pop("lrsg"),
                         dict(enabled=True, rank=4, beta=1.0, gate_init_logit=4.0))
        current["method"] = reference["method"]
        for key in ("gradient_transport", "low_rank_transport"):
            current["method_config"][key] = False
        self.assertEqual(current, reference)

    def test_classifier_gradients_preserved_across_inner_steps(self):
        transport = LowRankTransport(self.model, baseline.read_config()["lrsg"])
        original_grad = torch.autograd.grad
        previous = []
        updates = []
        def observe(loss, fast, **kwargs):
            self.assertEqual(len(fast), len(self.weights) + 2)
            self.assertEqual(fast[-2].shape, (2, 4))
            self.assertEqual(fast[-1].shape, (2,))
            if previous:
                for current, initial, grad in zip(fast[-2:], previous[-1], updates[-1]):
                    torch.testing.assert_close(current, initial - self.cfg["classifier_lr"] * grad)
            grads = original_grad(loss, fast, **kwargs)
            previous.append([w.detach().clone() for w in fast[-2:]])
            updates.append([g.clamp(-self.cfg["grad_clip"], self.cfg["grad_clip"])
                            for g in grads[-2:]])
            return grads
        with patch.object(torch.autograd, "grad", side_effect=observe):
            fast = helpers.adapt(self.model, self.weights, self.x, self.y,
                                 self.cfg, transport=transport)
        self.assertEqual(len(previous), self.cfg["inner_steps"])
        for current, initial, grad in zip(fast[-2:], previous[-1], updates[-1]):
            torch.testing.assert_close(current, initial - self.cfg["classifier_lr"] * grad)
        helpers.query_loss(self.model, fast, self.x + .2, self.y)[1].backward()
        self.assertTrue(all(p.grad is not None for p in transport.parameters()))

    def test_transport_order_mismatch_rejected(self):
        transport = LowRankTransport(self.model, baseline.read_config()["lrsg"])
        transport.names.reverse()
        with self.assertRaisesRegex(ValueError, "encoder ordering"):
            helpers.adapt(self.model, self.weights, self.x, self.y,
                          self.cfg, transport=transport)

    def test_transport_formula_conv_bias_initialization_and_rng(self):
        model = nn.Sequential(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), nn.Flatten(), nn.Linear(6, 3))
        rng = torch.random.get_rng_state().clone()
        transport = LowRankTransport(model, baseline.read_config()["lrsg"])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, p in model.named_parameters():
            key = transport.indices[name]
            grad = torch.randn_like(p)
            torch.testing.assert_close(transport.transport_gradient(name, grad), grad * torch.sigmoid(torch.tensor(4.)))
            self.assertEqual(key in transport.u, p.ndim >= 2)
            if key in transport.u:
                self.assertEqual(transport.u[key].shape, (p.shape[0], 4))
                self.assertEqual(transport.v[key].count_nonzero().item(), 0)
                with torch.no_grad():
                    transport.v[key].normal_()
                expected = (grad.reshape(p.shape[0], -1) * torch.sigmoid(torch.tensor(4.))
                            + transport.u[key] @ transport.v[key].T @ grad.reshape(p.shape[0], -1))
                torch.testing.assert_close(transport.transport_gradient(name, grad), expected.reshape_as(grad))
        alternate = LowRankTransport(model, dict(baseline.read_config()["lrsg"], rank=2, beta=.3))
        self.assertEqual(alternate.u["0"].shape, (6, 2))
        self.assertEqual(alternate.beta, .3)

    def test_query_gradients_outer_only_first_order(self):
        transport = LowRankTransport(self.model, baseline.read_config()["lrsg"])
        before = [p.detach().clone() for p in transport.parameters()]
        original_grad = torch.autograd.grad
        def observe(loss, fast, **kwargs):
            self.assertEqual(len(fast), len(self.weights) + 2)
            self.assertFalse(any(id(p) == id(w) for p in transport.parameters() for w in fast))
            self.assertFalse(kwargs["create_graph"])
            grads = original_grad(loss, fast, **kwargs)
            self.assertTrue(all(g.grad_fn is None and not g.requires_grad for g in grads))
            return grads
        with patch.object(torch.autograd, "grad", side_effect=observe):
            fast = helpers.adapt(self.model, self.weights, self.x, self.y, self.cfg, transport=transport)
        for a, b in zip(before, transport.parameters()):
            torch.testing.assert_close(a, b)
        _, loss = helpers.query_loss(self.model, fast, self.x + .3, self.y)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in transport.parameters()))
        self.assertGreater(sum(p.grad.abs().sum() for p in transport.logits.values()).item(), 0)
        self.assertGreater(sum(p.grad.abs().sum() for p in transport.v.values()).item(), 0)
        # V=0 makes the first U gradient exactly zero; it is connected, not detached.
        self.assertEqual(sum(p.grad.abs().sum() for p in transport.u.values()).item(), 0)
        torch.optim.SGD(transport.parameters(), lr=.1).step()
        transport.zero_grad()
        fast = helpers.adapt(self.model, self.weights, self.x, self.y, self.cfg, transport=transport)
        helpers.query_loss(self.model, fast, self.x + .3, self.y)[1].backward()
        self.assertGreater(sum(p.grad.abs().sum() for p in transport.u.values()).item(), 0)

    def test_disabled_equivalence_variable_way_and_eval(self):
        transport = LowRankTransport(self.model, dict(baseline.read_config()["lrsg"], enabled=False))
        self.assertEqual(list(transport.parameters()), [])
        expected = helpers.adapt(self.model, self.weights, self.x, self.y, self.cfg)
        actual = helpers.adapt(self.model, self.weights, self.x, self.y, self.cfg, transport=transport)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        transport = LowRankTransport(self.model, baseline.read_config()["lrsg"])
        with torch.no_grad():
            for p in transport.v.values():
                p.fill_(.5)
        for ways in (2, 7, 20):
            labels = torch.arange(ways).repeat(2)
            x = torch.randn(len(labels), 3)
            transport.train()
            trained = helpers.adapt(self.model, self.weights, x, labels, self.cfg, ways, transport)
            transport.eval()
            with torch.no_grad():
                evaluated = helpers.adapt(self.model, self.weights, x, labels, self.cfg, ways, transport)
            for a, b in zip(trained, evaluated):
                torch.testing.assert_close(a, b)

    def test_metrics_and_wandb(self):
        transport = LowRankTransport(self.model, baseline.read_config()["lrsg"])
        with torch.no_grad():
            transport.v["0"].fill_(.5)
        grad = torch.ones_like(self.weights[0])
        transport.transport_gradient("encoder.weight", grad)
        correction = transport.u["0"] @ transport.v["0"].T @ grad
        metrics = transport.metrics()
        self.assertAlmostEqual(metrics["lrsg/correction_to_gradient_ratio"],
                               (correction.norm() / grad.norm()).item())
        calls = []
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,
                wandb=SimpleNamespace(run=object(), log=lambda values: calls.append(values))):
            log_metrics(SimpleNamespace(logs_dir=directory), transport, 5)
            row = json.loads((Path(directory)/"lrsg_metrics.jsonl").read_text())
            self.assertEqual(row, calls[0])
            self.assertEqual(len(row), 9)
        self.assertEqual(transport._count, 0)

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
