"""CPU unit/synthetic checks; no Meta-Album data or full training is used."""
import copy
import importlib.util
import inspect
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
REFERENCE = ROOT.parent / "fo_proto_tclrsgmaml"
sys.path.insert(0, str(ROOT))
import model as baseline
import helpers_fo_proto_constz_lrsgmaml as helpers
from task_transport import ConstantConditionedTransport
from metrics import log_metrics


def reference_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ReferenceTransport = reference_module("tc_reference_transport", "task_transport.py").TaskConditionedTransport


class Tiny(nn.Module):
    in_features = 512

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 512)
        self.model = nn.ModuleDict({"out": nn.Identity()})

    def forward_weights(self, x, weights, embedding=False):
        features = F.linear(x, weights[0], weights[1]).tanh() / (512 ** .5)
        return features if embedding else F.linear(features, weights[-2], weights[-1])


class ConstantConditionTests(unittest.TestCase):
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
        self.transport = ConstantConditionedTransport(self.model, self.config)
        self.x = torch.randn(6, 3, dtype=torch.float64)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1])
        self.query = torch.randn(6, 3, dtype=torch.float64)

    def adapt(self, support=None):
        return helpers.adapt(self.model, self.weights, self.x if support is None else support,
                             self.y, self.cfg, 2, self.transport)

    def learned_state(self):
        with torch.no_grad():
            for u in self.transport.u.values():
                u.normal_(0, .2)
            self.transport.gate_net.out.weight.normal_(0, .1)
            self.transport.gate_net.hidden.bias.fill_(.1)

    def test_same_parameter_is_only_gate_input_across_support_and_query(self):
        self.learned_state()
        seen = []
        handle = self.transport.gate_net.register_forward_pre_hook(
            lambda module, args: seen.append(args[0]))
        try:
            for support, query in ((self.x, self.query), (self.x + 3, self.query * -2)):
                with patch.object(self.model, "forward_weights", wraps=self.model.forward_weights) as forward:
                    fast = self.adapt(support)
                    self.assertEqual(sum(c.kwargs.get("embedding", False)
                                         for c in forward.call_args_list), 1)
                count = len(seen)
                helpers.query_loss(self.model, fast, query, self.y)
                self.assertEqual(len(seen), count)
            self.assertEqual(len(seen), 2)
            self.assertTrue(all(value is self.transport.z for value in seen))
            self.assertEqual(tuple(seen[0].shape), (512,))
            self.assertTrue(seen[0].requires_grad)
            self.assertEqual(list(inspect.signature(self.transport.condition).parameters), [])
            with self.assertRaises(TypeError):
                self.transport.condition(self.x)
            # Conditioning has no graph connection to episode data or encoder.
            independent_support = self.x.clone().requires_grad_()
            independent_query = self.query.clone().requires_grad_()
            c = self.transport.condition()[1]["0"].sum()
            self.assertTrue(all(g is None for g in torch.autograd.grad(
                c, [independent_support, independent_query] + self.weights, allow_unused=True)))
        finally:
            handle.remove()

    def test_identical_gradient_has_bitwise_identical_transport_across_tasks(self):
        self.learned_state()
        conditions = []
        original = self.transport.condition
        def record():
            value = original()
            conditions.append(value)
            return value
        with patch.object(self.transport, "condition", side_effect=record):
            self.adapt(self.x)
            self.adapt(self.x - 7)
        self.assertEqual(len(conditions), 2)
        for name, p in self.model.named_parameters():
            gradient = torch.randn_like(p)
            a = self.transport.transport_gradient(name, gradient, conditions[0])
            b = self.transport.transport_gradient(name, gradient, conditions[1])
            self.assertTrue(torch.equal(a, b))

    def test_query_loss_has_nonzero_z_gradient_after_zero_layers_are_active(self):
        self.learned_state()
        loss = helpers.query_loss(self.model, self.adapt(), self.query, self.y)[1]
        loss.backward()
        self.assertIsNotNone(self.transport.z.grad)
        self.assertTrue(torch.isfinite(self.transport.z.grad).all())
        self.assertGreater(self.transport.z.grad.norm().item(), 0)
        for parameter in self.weights + list(self.transport.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_zero_initial_output_and_connected_zero_z_gradient(self):
        t = self.transport
        self.assertIsInstance(t.z, nn.Parameter)
        self.assertEqual(t.z.count_nonzero().item(), 0)
        self.assertEqual(t.gate_net.out.weight.count_nonzero().item(), 0)
        self.assertEqual(t.gate_net.out.bias.count_nonzero().item(), 0)
        a, c = t.condition()
        self.assertEqual(a.count_nonzero().item(), 0)
        self.assertTrue(all(value.count_nonzero().item() == 0 for value in c.values()))
        helpers.query_loss(self.model, self.adapt(), self.query, self.y)[1].backward()
        self.assertIsNotNone(t.z.grad)
        self.assertEqual(t.z.grad.count_nonzero().item(), 0)
        self.assertGreater(t.u["0"].grad.norm().item(), 0)
        self.assertEqual(t.v["0"].grad.count_nonzero().item(), 0)

    def test_outer_adam_activates_z_from_exact_default_initialization(self):
        parameters = self.weights + list(self.transport.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.cfg["outer_lr"])
        gradients = []
        initial = self.transport.z.detach().clone()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            helpers.query_loss(self.model, self.adapt(), self.query, self.y)[1].backward()
            gradients.append(self.transport.z.grad.norm().item())
            for p in parameters:
                p.grad.clamp_(-self.cfg["grad_clip"], self.cfg["grad_clip"])
            optimizer.step()
        self.assertEqual(gradients[0], 0)
        self.assertGreater(max(gradients[2:]), 0)
        self.assertFalse(torch.equal(initial, self.transport.z))

    def test_first_order_inner_loop_preserves_z_and_never_transports_head(self):
        self.learned_state()
        original = torch.autograd.grad
        previous = []
        def observe(loss, fast, **kwargs):
            self.assertFalse(kwargs["create_graph"])
            self.assertTrue(kwargs["retain_graph"])
            self.assertEqual(len(fast), len(self.weights) + 2)
            self.assertFalse({id(w) for w in fast} & {id(p) for p in self.transport.parameters()})
            if previous:
                for current, old, grad in zip(fast[-2:], *previous[-1]):
                    torch.testing.assert_close(current, old - self.cfg["classifier_lr"] * grad, rtol=0, atol=0)
            grads = original(loss, fast, **kwargs)
            self.assertTrue(all(g.grad_fn is None for g in grads))
            previous.append(([p.detach().clone() for p in fast[-2:]],
                             [g.clamp(-self.cfg["grad_clip"], self.cfg["grad_clip"]) for g in grads[-2:]]))
            return grads
        before = copy.deepcopy(self.transport.state_dict())
        with patch.object(torch.autograd, "grad", side_effect=observe), patch.object(
                self.transport, "transport_gradient", wraps=self.transport.transport_gradient) as transformed:
            fast = self.adapt()
        for current, old, grad in zip(fast[-2:], *previous[-1]):
            torch.testing.assert_close(current, old - self.cfg["classifier_lr"] * grad, rtol=0, atol=0)
        self.assertEqual([call.args[0] for call in transformed.call_args_list],
                         self.transport.names * self.cfg["inner_steps"])
        for call in transformed.call_args_list:
            self.assertLessEqual(call.args[1].abs().max().item(), self.cfg["grad_clip"])
        for name, value in before.items():
            self.assertTrue(torch.equal(value, self.transport.state_dict()[name]))
        self.assertIsNone(self.transport.z.grad)

    def test_reference_gatenet_layout_initialization_and_rng_are_identical(self):
        state = torch.random.get_rng_state().clone()
        ref = ReferenceTransport(self.model, self.config)
        current = ConstantConditionedTransport(self.model, self.config)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(ref.gate_net.config(), current.gate_net.config())
        self.assertEqual(str(ref.gate_net), str(current.gate_net))
        self.assertEqual(ref.rank_layout, current.rank_layout)
        self.assertEqual(ref.output_size, current.output_size)
        architecture = current.architecture()
        architecture.pop("constant_condition")
        self.assertEqual(ref.architecture(), architecture)
        self.assertEqual(set(current.state_dict()) - set(ref.state_dict()), {"z"})
        for name, value in ref.state_dict().items():
            self.assertTrue(torch.equal(value, current.state_dict()[name]), name)
        for name in ("gate_net.py", "low_rank_transport.py", "network.py", "api.py",
                     "metrics.py", "weight_names.py", "metadata", "test.py"):
            self.assertEqual((ROOT / name).read_bytes(), (REFERENCE / name).read_bytes(), name)

    def test_conv_formula_layout_and_scalar_residual_disabled(self):
        encoder = nn.Sequential(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), nn.Linear(6, 2)).double()
        encoder.in_features = 512
        t = ConstantConditionedTransport(encoder, self.config)
        self.assertEqual([(r["name"], r["rank"], r["start"], r["stop"])
                          for r in t.rank_layout], [("0.weight", 4, 6, 10), ("2.weight", 2, 10, 12)])
        with torch.no_grad():
            t.gate_net.out.bias.copy_(torch.arange(12, dtype=torch.float64) / 10)
            for u in t.u.values():
                u.normal_()
        a, c = t.condition()
        self.assertEqual(a.count_nonzero().item(), 0)
        torch.testing.assert_close(c["0"], torch.arange(6, 10, dtype=torch.float64) / 10)
        ref = ReferenceTransport(encoder, self.config)
        ref.load_state_dict({k: v for k, v in t.state_dict().items() if k != "z"})
        ref_condition = ref.condition(t.z.detach())
        for name, p in encoder.named_parameters():
            key = t.indices[name]
            g = torch.randn_like(p)
            expected = t.logits[key].sigmoid() * g
            if key in t.u:
                matrix = g.reshape(g.shape[0], -1)
                expected = expected + (t.beta * t.u[key] @ torch.diag(1 + c[key]) @ t.v[key].T @ matrix).reshape_as(g)
            actual = t.transport_gradient(name, g, (a, c))
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.equal(actual, ref.transport_gradient(name, g, ref_condition)))
        self.learned_state()
        helpers.query_loss(self.model, self.adapt(), self.query, self.y)[1].backward()
        n = len(self.transport.names)
        self.assertEqual(self.transport.gate_net.out.weight.grad[:n].count_nonzero().item(), 0)
        self.assertEqual(self.transport.gate_net.out.bias.grad[:n].count_nonzero().item(), 0)
        self.assertGreater(self.transport.gate_net.out.weight.grad[n:].norm().item(), 0)
        self.assertTrue(all(p.grad.abs().item() > 0 for p in self.transport.logits.values()))

    def test_config_parity_and_only_zero_initialization_supported(self):
        baseline.validate_config(self.config)
        reference = json.loads((REFERENCE / "config.json").read_text())
        adjusted = copy.deepcopy(self.config)
        self.assertEqual(adjusted.pop("constant_condition"), dict(enabled=True, init="zero"))
        adjusted["method"] = reference["method"]
        reference["task_conditioning"]["scalar_delta_scale"] = 0.0
        self.assertEqual(adjusted, reference)
        for section, key, value in (("constant_condition", "init", "mean_support"),
                                    ("constant_condition", "enabled", False),
                                    ("task_conditioning", "scalar_delta_scale", 1),
                                    ("task_conditioning", "input_norm", "l2"),
                                    ("method_config", "first_order", False)):
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.assertRaises(ValueError):
                baseline.validate_config(config)

    def test_synthetic_meta_fit_optimizer_clipping_and_checkpoint_roundtrip(self):
        config = copy.deepcopy(self.config)
        config["experiment_config"] = dict(train_iterations=6, validation_tasks=1, validate_every=2)
        x = self.x.float()
        task = SimpleNamespace(num_ways=2, support_set=(x, self.y, self.y),
                               query_set=(self.query.float(), self.y, self.y))
        generator = lambda count: iter([task] * count)
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=config), patch.object(torch.cuda, "is_available", return_value=False):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))
            optimized = [p for group in meta.optimizer.param_groups for p in group["params"]]
            self.assertEqual({id(p) for p in optimized}, {id(p) for p in meta.weights + list(meta.transport.parameters())})
            self.assertEqual(sum(p is meta.transport.z for p in optimized), 1)
            self.assertFalse(any(p is meta.transport.z for p in meta.weights))
            self.assertEqual(meta.optimizer.param_groups[0]["lr"], self.cfg["outer_lr"])
            learner = meta.meta_fit(generator, generator)
            self.assertEqual(meta.optimizer.state[meta.transport.z]["step"].item(), 3)
            self.assertGreater(meta.transport.z.norm().item(), 0)
            # Give the saved best state a nonzero z, including when best selection
            # happened before z first updated, to exercise strict persistence.
            with torch.no_grad():
                learner.transport.z.fill_(.125)
            learner.state = baseline.snapshot(learner.learner, learner.transport)
            support = (x, self.y, self.y, 2, 3)
            expected = learner.fit(support).predict(x)
            with tempfile.TemporaryDirectory() as directory:
                learner.save(directory)
                restored = baseline.MyLearner()
                restored.load(directory)
                self.assertTrue(torch.equal(learner.transport.z, restored.transport.z))
                torch.testing.assert_close(torch.tensor(expected), torch.tensor(restored.fit(support).predict(x)), rtol=0, atol=0)
                data = torch.load(Path(directory) / "max-va.pth", weights_only=True)
                self.assertIn("z", data["state"]["lrsg"])
                data["state"]["architecture"]["constant_condition"]["shape"] = [4]
                torch.save(data, Path(directory) / "max-va.pth")
                with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                    baseline.MyLearner().load(directory)
                data["state"]["architecture"]["constant_condition"]["shape"] = [512]
                del data["state"]["lrsg"]["z"]
                torch.save(data, Path(directory) / "max-va.pth")
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    baseline.MyLearner().load(directory)

    def test_variable_way_eval_preserves_z(self):
        self.learned_state()
        self.transport.eval()
        before = self.transport.z.detach().clone()
        for ways in (2, 7, 20):
            labels = torch.arange(ways).repeat(2)
            support = torch.randn(len(labels), 3, dtype=torch.float64)
            with torch.no_grad():
                fast = helpers.adapt(self.model, self.weights, support, labels, self.cfg, ways, self.transport)
                out, loss = helpers.query_loss(self.model, fast, support, labels)
            self.assertEqual(out.shape, (len(labels), ways))
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.equal(before, self.transport.z))
        self.assertEqual(self.transport._task_count, 0)

    def test_constant_condition_metrics_and_logging(self):
        self.learned_state()
        self.transport.condition()
        self.transport.condition()
        metrics = self.transport.metrics()
        self.assertEqual(metrics["tc/scalar_delta_abs_mean"], 0)
        self.assertEqual(metrics["tc/low_rank_delta_task_std_mean"], 0)
        calls = []
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, wandb=SimpleNamespace(run=object(), log=calls.append)):
            log_metrics(SimpleNamespace(logs_dir=directory), self.transport, 5)
            self.assertEqual(json.loads((Path(directory) / "lrsg_metrics.jsonl").read_text()), calls[0])
        self.assertEqual(self.transport._task_count, 0)

    def test_real_resnet_synthetic_five_step_backward_and_layout(self):
        encoder = baseline.make_encoder(dict(num_classes=2, dev=torch.device("cpu"), num_blocks=18, pretrained=False))
        t = ConstantConditionedTransport(encoder, self.config)
        reference = ReferenceTransport(encoder, self.config)
        self.assertEqual(t.gate_net.config(), reference.gate_net.config())
        self.assertEqual(t.rank_layout, reference.rank_layout)
        self.assertEqual(tuple(t.z.shape), (512,))
        x = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 0, 1])
        fast = helpers.adapt(encoder, list(encoder.parameters()), x, labels, self.cfg, 2, t)
        helpers.query_loss(encoder, fast, x + .1, labels)[1].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in list(encoder.parameters()) + list(t.parameters())))


if __name__ == "__main__":
    unittest.main()
