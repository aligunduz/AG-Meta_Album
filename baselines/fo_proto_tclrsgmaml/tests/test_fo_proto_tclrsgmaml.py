"""CPU-only numerical ablation and submission regression checks."""
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
sys.path.insert(0, str(ROOT))
import model as baseline
import helpers_fo_proto_tclrsgmaml as helpers
from task_transport import TaskConditionedTransport
from metrics import log_metrics


def reference_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT.parent / "fo_proto_lrsgmaml" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference_helpers = reference_module("reference_helpers", "helpers_fo_proto_lrsgmaml.py")
ReferenceTransport = reference_module("reference_transport", "low_rank_transport.py").LowRankTransport


class Tiny(nn.Module):
    in_features = 4

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.model = nn.ModuleDict({"out": nn.Identity()})

    def forward_weights(self, x, weights, embedding=False):
        features = F.linear(x, weights[0], weights[1]).tanh()
        return features if embedding else F.linear(features, weights[-2], weights[-1])


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(17)
        self.model = Tiny().double()
        self.weights = list(self.model.parameters())
        self.config = baseline.read_config()
        self.cfg = self.config["method_config"]
        self.transport = TaskConditionedTransport(self.model, self.config)
        self.x = torch.randn(6, 3, dtype=torch.float64)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1])
        self.query = torch.randn(6, 3, dtype=torch.float64)

    def adapt(self, transport=None, **kwargs):
        return helpers.adapt(self.model, self.weights, self.x, self.y, self.cfg,
                             transport=self.transport if transport is None else transport, **kwargs)

    def test_initialization_rank_rng_and_no_normalization(self):
        state = torch.random.get_rng_state().clone()
        reference = ReferenceTransport(self.model, self.config["lrsg"])
        current = TaskConditionedTransport(self.model, self.config)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, current.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(current.u["0"].shape, (4, 4))
        self.assertEqual(current.u["0"].count_nonzero(), 0)
        self.assertFalse(torch.allclose(current.v["0"].norm(dim=0), torch.ones(4, dtype=torch.float64)))
        tiny = nn.Sequential(nn.Linear(3, 2))
        tiny.in_features = 2
        small = TaskConditionedTransport(tiny, self.config)
        self.assertEqual(small.u["0"].shape, (2, 2))
        self.assertEqual(small.rank_layout[0]["rank"], 2)
        with torch.no_grad():
            current.u["0"].fill_(7)
            current.v["0"].fill_(11)
        before = copy.deepcopy(current.state_dict())
        current.transport_gradient("encoder.weight", torch.ones_like(self.weights[0]),
                                   current.condition(torch.ones(4, dtype=torch.float64)))
        for key, value in before.items():
            torch.testing.assert_close(value, current.state_dict()[key], rtol=0, atol=0)

    def test_zero_residual_equivalence_float64(self):
        reference = ReferenceTransport(self.model, self.config["lrsg"])
        # Check both initialization and a learned, nonzero low-rank operator;
        # U=0 alone would hide an incorrect coefficient of delta_c instead of 1+delta_c.
        for learned in (False, True):
            if learned:
                with torch.no_grad():
                    reference.u["0"].normal_(0, .2)
                self.transport.load_state_dict(reference.state_dict(), strict=False)
            actual = self.adapt()
            expected = reference_helpers.adapt(self.model, self.weights, self.x, self.y,
                                                self.cfg, transport=reference)
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            actual_logits, actual_loss = helpers.query_loss(self.model, actual, self.query, self.y)
            expected_logits, expected_loss = reference_helpers.query_loss(self.model, expected, self.query, self.y)
            torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
            a_grads = torch.autograd.grad(actual_loss, self.weights)
            b_grads = torch.autograd.grad(expected_loss, self.weights)
            for a, b in zip(a_grads, b_grads, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        residuals = self.transport.condition(torch.randn(4, dtype=torch.float64))
        self.assertEqual(residuals[0].count_nonzero(), 0)
        self.assertEqual(residuals[1]["0"].count_nonzero(), 0)

    def test_conv_formula_and_deterministic_mapping(self):
        encoder = nn.Sequential(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), nn.Linear(6, 2)).double()
        encoder.in_features = 2
        config = copy.deepcopy(self.config)
        config["task_conditioning"].update(scalar_delta_scale=.3, low_rank_delta_scale=.7)
        transport = TaskConditionedTransport(encoder, config)
        self.assertEqual([(r["name"], r["rank"], r["start"], r["stop"])
                          for r in transport.rank_layout], [("0.weight", 4, 6, 10), ("2.weight", 2, 10, 12)])
        with torch.no_grad():
            transport.gate_net.out.bias.copy_(torch.arange(12, dtype=torch.float64) / 10)
            for p in transport.u.values():
                p.normal_()
        delta_a, delta_c = transport.condition(torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(delta_a, torch.arange(6, dtype=torch.float64) * .03)
        torch.testing.assert_close(delta_c["0"], torch.arange(6, 10, dtype=torch.float64) * .07)
        for name, p in encoder.named_parameters():
            key = transport.indices[name]
            grad = torch.randn_like(p)
            expected = (transport.logits[key] + delta_a[int(key)]).sigmoid() * grad
            if key in transport.u:
                matrix = grad.reshape(grad.shape[0], -1)
                correction = transport.beta * transport.u[key] @ torch.diag(1 + delta_c[key]) @ transport.v[key].T @ matrix
                expected = expected + correction.reshape_as(grad)
            else:
                self.assertEqual(p.ndim, 1)
            torch.testing.assert_close(transport.transport_gradient(name, grad, (delta_a, delta_c)), expected)

    def test_scalar_and_rank_residuals_are_separate(self):
        t = self.transport
        with torch.no_grad():
            t.u["0"].normal_()
        grad = torch.randn_like(self.weights[0])
        zeros = torch.zeros(2, dtype=torch.float64)
        base = t.transport_gradient("encoder.weight", grad, (zeros, {"0": torch.zeros(4, dtype=torch.float64)}))
        scalar = t.transport_gradient("encoder.weight", grad, (zeros + .5, {"0": torch.zeros(4, dtype=torch.float64)}))
        torch.testing.assert_close(scalar - base, ((t.logits["0"] + .5).sigmoid() - t.logits["0"].sigmoid()) * grad)
        delta = torch.tensor([.1, -.5, 1., -2.], dtype=torch.float64)
        ranked = t.transport_gradient("encoder.weight", grad, (zeros, {"0": delta}))
        torch.testing.assert_close(ranked - base, t.beta * t.u["0"] @ torch.diag(delta) @ t.v["0"].T @ grad)

    def test_support_only_detach_single_forward_and_episode_isolation(self):
        with torch.no_grad():
            self.transport.gate_net.out.weight.normal_(0, .1)
        seen = []
        def record(module, args):
            seen.append(args[0])
        handle = self.transport.gate_net.register_forward_pre_hook(record)
        try:
            with patch.object(self.model, "forward_weights", wraps=self.model.forward_weights) as forward:
                first = self.adapt()
                self.assertEqual(sum(call.kwargs.get("embedding", False) for call in forward.call_args_list), 1)
            self.assertEqual(len(seen), 1)
            self.assertFalse(seen[0].requires_grad)
            expected = self.model.forward_weights(self.x, self.weights, embedding=True).mean(0).detach()
            torch.testing.assert_close(seen[0], expected)
            self.assertTrue(all(g is None for g in torch.autograd.grad(
                self.transport.condition(expected)[0].sum(), self.weights, allow_unused=True)))
            helpers.adapt(self.model, self.weights, self.x + 1, self.y, self.cfg, transport=self.transport)
            again = self.adapt()
            for a, b in zip(first, again):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            query = self.query.clone().requires_grad_()
            self.assertIsNone(torch.autograd.grad(first[-2].sum(), query, allow_unused=True)[0])
        finally:
            handle.remove()

    def test_fo_outer_gradients_fast_weights_and_ungated_head(self):
        # Nonzero state exposes all paths; zero-initialized U/final head have
        # expected connected zero gradients for V/rank branch/hidden at step 0.
        with torch.no_grad():
            self.transport.u["0"].normal_(0, .2)
            self.transport.gate_net.out.weight.normal_(0, .1)
        original = torch.autograd.grad
        previous = []
        def observe(loss, fast, **kwargs):
            self.assertFalse(kwargs["create_graph"])
            self.assertTrue(kwargs["retain_graph"])
            self.assertEqual(len(fast), len(self.weights) + 2)
            self.assertFalse({id(w) for w in fast} & {id(p) for p in self.transport.parameters()})
            if previous:
                for current, old, grad in zip(fast[-2:], *previous[-1]):
                    torch.testing.assert_close(current, old - self.cfg["classifier_lr"] * grad)
            grads = original(loss, fast, **kwargs)
            self.assertTrue(all(g.grad_fn is None for g in grads))
            previous.append(([p.detach().clone() for p in fast[-2:]],
                             [g.clamp(-self.cfg["grad_clip"], self.cfg["grad_clip"]) for g in grads[-2:]]))
            return grads
        before = copy.deepcopy(self.transport.state_dict())
        with patch.object(torch.autograd, "grad", side_effect=observe):
            fast = self.adapt()
        for current, old, grad in zip(fast[-2:], *previous[-1]):
            torch.testing.assert_close(current, old - self.cfg["classifier_lr"] * grad)
        for key, value in before.items():
            torch.testing.assert_close(value, self.transport.state_dict()[key], rtol=0, atol=0)
        helpers.query_loss(self.model, fast, self.query, self.y)[1].backward()
        for p in self.weights + list(self.transport.parameters()):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.norm().item(), 0)

    def test_initial_query_gradients_connected(self):
        helpers.query_loss(self.model, self.adapt(), self.query, self.y)[1].backward()
        self.assertTrue(all(p.grad is not None for p in self.transport.parameters()))
        self.assertGreater(self.transport.u["0"].grad.norm().item(), 0)
        self.assertEqual(self.transport.v["0"].grad.norm().item(), 0)
        self.assertGreater(self.transport.gate_net.out.weight.grad[:2].norm().item(), 0)
        torch.optim.SGD(self.transport.parameters(), lr=.1).step()
        self.transport.zero_grad()
        helpers.query_loss(self.model, self.adapt(), self.query, self.y)[1].backward()
        self.assertGreater(self.transport.v["0"].grad.norm().item(), 0)
        self.assertGreater(self.transport.gate_net.out.weight.grad[2:].norm().item(), 0)

    def test_variable_way_no_grad_eval_uses_learned_conditioning(self):
        with torch.no_grad():
            self.transport.gate_net.out.weight.normal_(0, .2)
            self.transport.u["0"].normal_(0, .2)
        self.transport.eval()
        for ways in (2, 7, 20):
            labels = torch.arange(ways).repeat(2)
            support = torch.randn(len(labels), 3, dtype=torch.float64)
            with torch.no_grad():
                fast = helpers.adapt(self.model, self.weights, support, labels, self.cfg, ways, self.transport)
                out, _ = helpers.query_loss(self.model, fast, support, labels)
            self.assertEqual(out.shape, (len(labels), ways))
            reference = ReferenceTransport(self.model, self.config["lrsg"])
            reference.load_state_dict({k: v for k, v in self.transport.state_dict().items() if not k.startswith("gate_net.")})
            unconditioned = reference_helpers.adapt(self.model, self.weights, support, labels, self.cfg, ways, reference)
            self.assertFalse(torch.equal(fast[0], unconditioned[0]))
        self.assertEqual(self.transport._task_count, 0)

    def test_disabled_conditioning_matches_lrsg(self):
        config = copy.deepcopy(self.config)
        config["task_conditioning"]["enabled"] = False
        transport = TaskConditionedTransport(self.model, config)
        self.assertIsNone(transport.gate_net)
        reference = ReferenceTransport(self.model, config["lrsg"])
        for a, b in zip(self.adapt(transport), reference_helpers.adapt(
                self.model, self.weights, self.x, self.y, self.cfg, transport=reference)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_config_parity(self):
        reference = json.loads((ROOT.parent / "fo_proto_lrsgmaml/config.json").read_text())
        config = copy.deepcopy(self.config)
        config.pop("task_conditioning")
        config["method"] = reference["method"]
        config["method_config"]["task_conditioned_gate"] = False
        self.assertEqual(reference, config)

    def test_train_checkpoint_optimizer_and_architecture(self):
        config = copy.deepcopy(self.config)
        config["experiment_config"] = dict(train_iterations=4, validation_tasks=1, validate_every=2)
        x = self.x.float()
        task = SimpleNamespace(num_ways=2, support_set=(x, self.y, self.y), query_set=(x + .2, self.y, self.y))
        generator = lambda count: iter([task] * count)
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=config), patch.object(torch.cuda, "is_available", return_value=False):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))
            ids = {id(p) for group in meta.optimizer.param_groups for p in group["params"]}
            self.assertEqual(ids, {id(p) for p in meta.weights + list(meta.transport.parameters())})
            learner = meta.meta_fit(generator, generator)
            support = (x, self.y, self.y, 2, 3)
            expected = learner.fit(support).predict(x)
            with tempfile.TemporaryDirectory() as directory:
                learner.save(directory)
                restored = baseline.MyLearner()
                restored.load(directory)
                torch.testing.assert_close(torch.tensor(expected), torch.tensor(restored.fit(support).predict(x)), rtol=0, atol=0)
                for key, value in learner.transport.state_dict().items():
                    torch.testing.assert_close(value, restored.transport.state_dict()[key], rtol=0, atol=0)
                data = torch.load(Path(directory)/"max-va.pth", weights_only=True)
                self.assertTrue(any(k.startswith("gate_net.") for k in data["state"]["lrsg"]))
                data["state"]["architecture"]["task_conditioning"]["rank_layout"][0]["start"] += 1
                torch.save(data, Path(directory)/"max-va.pth")
                with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                    baseline.MyLearner().load(directory)
            best = copy.deepcopy(meta.best_state)
            meta.best_score = 2
            with torch.no_grad():
                meta.transport.gate_net.out.bias.add_(1)
            meta.meta_valid(generator)
            for key, value in best["lrsg"].items():
                torch.testing.assert_close(value, meta.best_state["lrsg"][key], rtol=0, atol=0)

    def test_metrics_task_variation_and_logging(self):
        with torch.no_grad():
            self.transport.gate_net.out.weight.normal_(0, .2)
        a = self.transport.condition(torch.zeros(4, dtype=torch.float64))
        b = self.transport.condition(torch.ones(4, dtype=torch.float64))
        metrics = self.transport.metrics()
        expected = torch.stack((a[0], b[0])).std(dim=0, unbiased=False).mean().item()
        self.assertAlmostEqual(metrics["tc/scalar_delta_task_std_mean"], expected)
        self.assertEqual(metrics["tc/low_rank_coeff_mean"], 1 + metrics["tc/low_rank_delta_mean"])
        calls = []
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, wandb=SimpleNamespace(run=object(), log=calls.append)):
            log_metrics(SimpleNamespace(logs_dir=directory), self.transport, 5)
            self.assertEqual(json.loads((Path(directory)/"lrsg_metrics.jsonl").read_text()), calls[0])
        self.assertEqual(self.transport._task_count, 0)

    def test_real_resnet_five_steps_backward(self):
        encoder = baseline.make_encoder(dict(num_classes=2, dev=torch.device("cpu"), num_blocks=18, pretrained=False))
        transport = TaskConditionedTransport(encoder, self.config)
        x = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 0, 1])
        fast = helpers.adapt(encoder, list(encoder.parameters()), x, labels, self.cfg, 2, transport)
        helpers.query_loss(encoder, fast, x + .1, labels)[1].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in list(encoder.parameters()) + list(transport.parameters())))


if __name__ == "__main__":
    unittest.main()
