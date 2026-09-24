"""CPU synthetic checks; no dataset download or full training."""
import copy
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "fo_proto_constz_lrsgmaml"
sys.path.insert(0, str(ROOT))
import model as baseline
import helpers_fo_proto_constz_lrsgmaml as helpers
from learned_inner_lr import LearnedInnerLR
from task_transport import ConstantConditionedTransport
from metrics import log_metrics

spec = importlib.util.spec_from_file_location("original_ema_helpers", SOURCE / "helpers_fo_proto_constz_lrsgmaml.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class Tiny(nn.Module):
    in_features = 512

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 512)
        self.model = nn.ModuleDict({"out": nn.Identity()})

    def forward_weights(self, x, weights, embedding=False):
        z = F.linear(x, weights[0], weights[1]).tanh() / 512 ** .5
        return z if embedding else F.linear(z, weights[-2], weights[-1])


class LearnedLRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(94)
        self.config = baseline.read_config()
        self.encoder = Tiny().double()
        self.transport = ConstantConditionedTransport(self.encoder, self.config)
        self.lrs = LearnedInnerLR(self.encoder, 5)
        self.x, self.q = torch.randn(2, 6, 3, dtype=torch.float64)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1])

    def adapt(self, lrs=True):
        return helpers.adapt(self.encoder, list(self.encoder.parameters()), self.x,
                             self.y, self.config["method_config"], 2, self.transport,
                             inner_lrs=self.lrs if lrs else None)

    def test_initialization_matches_original_ema_adaptation_and_outer_gradients(self):
        for dtype in (torch.float32, torch.float64):
            self.encoder.to(dtype=dtype)
            self.transport.to(dtype=dtype)
            self.lrs = LearnedInnerLR(self.encoder, 5)
            self.x, self.q = self.x.to(dtype), self.q.to(dtype)
            torch.testing.assert_close(self.lrs.rates(), torch.full_like(self.lrs.logits, .01))
            for initialized in (False, True):
                if initialized:
                    self.transport.complete_training_episode(torch.randn_like(self.transport.m))
                    with torch.no_grad():
                        for u in self.transport.u.values():
                            u.normal_(0, .1)
                        self.transport.gate_net.out.weight.normal_(0, .1)
                before = copy.deepcopy(self.transport.state_dict())
                fast = self.adapt()
                reference = original.adapt(self.encoder, list(self.encoder.parameters()), self.x,
                                          self.y, self.config["method_config"], 2, self.transport)
                for actual, expected in zip(fast, reference):
                    torch.testing.assert_close(actual, expected)
                parameters = list(self.encoder.parameters()) + list(self.transport.parameters())
                g1 = torch.autograd.grad(helpers.query_loss(self.encoder, fast, self.q, self.y)[1], parameters)
                g2 = torch.autograd.grad(original.query_loss(self.encoder, reference, self.q, self.y)[1], parameters)
                for actual, expected in zip(g1, g2):
                    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
                for key in before:
                    self.assertTrue(torch.equal(before[key], self.transport.state_dict()[key]))

    def test_every_step_and_tensor_receives_query_gradient_and_adam_updates(self):
        self.transport.complete_training_episode(torch.randn_like(self.transport.m))
        before = self.lrs.logits.detach().clone()
        optimizer = torch.optim.Adam(self.lrs.parameters(), lr=.001)
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            fast = self.adapt()
        self.assertEqual(grad.call_count, 5)
        self.assertTrue(all(c.kwargs["create_graph"] is False for c in grad.call_args_list))
        helpers.query_loss(self.encoder, fast, self.q, self.y)[1].backward()
        self.assertEqual(self.lrs.logits.grad.shape, (5, 2))
        self.assertTrue(torch.isfinite(self.lrs.logits.grad).all())
        self.assertTrue((self.lrs.logits.grad.abs() > 0).all())
        optimizer.step()
        self.assertTrue((self.lrs.logits != before).all())

    def test_bounds_layout_and_rng(self):
        rng = torch.get_rng_state().clone()
        LearnedInnerLR(self.encoder, 5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with torch.no_grad():
            self.lrs.logits.copy_(torch.linspace(-100, 100, 10).reshape(5, 2))
        self.assertTrue((self.lrs.rates() >= .001).all())
        self.assertTrue((self.lrs.rates() <= .05).all())
        with self.assertRaises(ValueError):
            self.lrs.validate_encoder(self.encoder, 4)

    def test_checkpoint_restores_learned_rates_predictions_and_ema(self):
        with torch.no_grad():
            self.lrs.logits.add_(torch.arange(10).reshape(5, 2) / 10)
        self.transport.complete_training_episode(torch.randn_like(self.transport.m))
        state = baseline.snapshot(self.encoder, self.transport, self.lrs)
        args = dict(dev="cpu")
        with patch.object(baseline, "make_encoder", side_effect=lambda args: Tiny().double()), patch.object(torch.cuda, "is_available", return_value=False), tempfile.TemporaryDirectory() as directory:
            learner = baseline.MyLearner(args, state, self.config, .75)
            prediction = learner.fit((self.x, self.y, self.y, 2, 3)).predict(self.q)
            learner.save(directory)
            restored = baseline.MyLearner()
            restored.load(directory)
            self.assertTrue(torch.equal(restored.inner_lrs.logits, self.lrs.logits))
            self.assertTrue(torch.equal(restored.inner_lrs.rates(), self.lrs.rates()))
            self.assertEqual(restored.best_score, .75)
            actual = restored.fit((self.x, self.y, self.y, 2, 3)).predict(self.q)
            self.assertTrue((actual == prediction).all())
            for key, value in state["lrsg"].items():
                self.assertTrue(torch.equal(restored.transport.state_dict()[key], value))
            data = torch.load(Path(directory) / "max-va.pth", weights_only=True)
            data["state"]["inner_lr_layout"]["names"].reverse()
            torch.save(data, Path(directory) / "max-va.pth")
            with self.assertRaisesRegex(ValueError, "layout mismatch"):
                baseline.MyLearner().load(directory)

    def test_synthetic_training_uses_best_lr_snapshot_and_same_optimizer_protocol(self):
        config = copy.deepcopy(self.config)
        config["experiment_config"] = dict(train_iterations=4, validation_tasks=1, validate_every=2)
        x, q = self.x.float(), self.q.float()
        task = SimpleNamespace(num_ways=2, support_set=(x, self.y, self.y), query_set=(q, self.y, self.y))
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=config), patch.object(torch.cuda, "is_available", return_value=False):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))
            initial = meta.inner_lrs.logits.detach().clone()
            self.assertIsInstance(meta.optimizer, torch.optim.Adam)
            self.assertEqual(meta.optimizer.param_groups[0]["lr"], .001)
            self.assertTrue(any(p is meta.inner_lrs.logits for p in meta.meta_parameters))
            query = baseline.query_loss
            def controlled_query(*args):
                out, loss = query(*args)
                if not meta.transport.training:
                    # First validation is perfect; second is deliberately wrong.
                    out = F.one_hot(self.y if meta.transport.ema_completed_tasks.item() == 2 else 1-self.y, 2).float()
                return out, loss
            with patch.object(meta.optimizer, "step", wraps=meta.optimizer.step) as step, patch.object(baseline, "query_loss", side_effect=controlled_query):
                learner = meta.meta_fit(lambda n: iter([task] * n), lambda n: iter([task] * n))
            self.assertEqual(step.call_count, 2)
            self.assertEqual(meta.transport.ema_completed_tasks.item(), 4)
            self.assertEqual(learner.transport.ema_completed_tasks.item(), 2)
            self.assertEqual(learner.best_score, 1.)
            self.assertFalse(torch.equal(initial, learner.inner_lrs.logits))
            self.assertTrue(torch.equal(learner.inner_lrs.logits, meta.best_state["inner_lrs"]["logits"]))
            self.assertFalse(torch.equal(learner.inner_lrs.logits, meta.inner_lrs.logits))

    def test_real_resnet_five_step_query_backward(self):
        encoder = baseline.make_encoder(dict(num_classes=2, dev=torch.device("cpu"), num_blocks=18, pretrained=False))
        transport = ConstantConditionedTransport(encoder, self.config)
        lrs = LearnedInnerLR(encoder, 5)
        x = torch.randn(8, 3, 64, 64)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        fast = helpers.adapt(encoder, list(encoder.parameters()), x, labels,
                             self.config["method_config"], 2, transport, inner_lrs=lrs)
        helpers.query_loss(encoder, fast, torch.randn_like(x), labels)[1].backward()
        self.assertEqual(lrs.logits.shape, (5, len(list(encoder.parameters()))))
        for p in list(encoder.parameters()) + list(transport.parameters()) + list(lrs.parameters()):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue((lrs.logits.grad.abs().sum(dim=1) > 0).all())

    def test_log_values_and_original_config_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            log_metrics(SimpleNamespace(logs_dir=directory), self.transport, 5000, self.lrs)
            record = json.loads((Path(directory) / "lrsg_metrics.jsonl").read_text())
        for key, value in self.lrs.metrics().items():
            self.assertEqual(record[key], value)
        self.assertEqual(len(self.lrs.metrics()), 8)
        original_config = json.loads((SOURCE / "config.json").read_text())
        config = copy.deepcopy(self.config)
        del config["learned_encoder_lr"]
        config["method"] = original_config["method"]
        self.assertEqual(config, original_config)
        for name in ("task_transport.py", "low_rank_transport.py", "gate_net.py", "network.py", "test.py", "api.py", "weight_names.py"):
            self.assertEqual((ROOT / name).read_bytes(), (SOURCE / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
