"""CPU/synthetic regression checks for the independent global coefficient baseline."""
import ast
import copy
from contextlib import redirect_stdout
import importlib.util
import inspect
import io
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
REFERENCE = ROOT.parent / "fo_proto_lrsgmaml"
sys.path.insert(0, str(ROOT))
import model as baseline
import helpers_fo_proto_global_lrsgmaml as helpers
from low_rank_transport import GlobalLowRankTransport
from metrics import log_metrics


def load_reference(name, filename):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ReferenceTransport = load_reference("reference_lrsg_transport", "low_rank_transport.py").LowRankTransport
reference_helpers = load_reference("reference_lrsg_helpers", "helpers_fo_proto_lrsgmaml.py")


class Tiny(nn.Module):
    # Deliberately no in_features/task embedding dimension metadata.
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.model = nn.ModuleDict({"out": nn.Identity()})

    def forward_weights(self, x, weights, embedding=False):
        features = F.linear(x, weights[0], weights[1]).tanh()
        return features if embedding else F.linear(features, weights[-2], weights[-1])


class GlobalCoefficientTests(unittest.TestCase):
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
        self.transport = GlobalLowRankTransport(self.model, self.config["lrsg"])
        self.x = torch.randn(6, 3, dtype=torch.float64)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1])
        self.query = torch.randn(6, 3, dtype=torch.float64)

    def adapt(self, transport=None, cfg=None):
        return helpers.adapt(self.model, self.weights, self.x, self.y,
                             self.cfg if cfg is None else cfg, 2,
                             self.transport if transport is None else transport)

    def test_config_and_unchanged_support_files_match_lrsg(self):
        current = baseline.read_config()
        reference = json.loads((REFERENCE / "config.json").read_text())
        self.assertEqual(current["method"], "fo-proto-global-lrsgmaml")
        current["method"] = reference["method"]
        self.assertEqual(current, reference)
        for name in ("api.py", "network.py", "weight_names.py", "metadata", "metrics.py", "test.py"):
            self.assertEqual((ROOT / name).read_bytes(), (REFERENCE / name).read_bytes())
        self.assertEqual((ROOT / "helpers_fo_proto_global_lrsgmaml.py").read_bytes(),
                         (REFERENCE / "helpers_fo_proto_lrsgmaml.py").read_bytes())
        for key, value in (("task_conditioned_gate", True), ("first_order", False)):
            invalid = baseline.read_config()
            invalid["method_config"][key] = value
            with self.assertRaises(ValueError):
                baseline.validate_config(invalid)

    def test_initialization_rank_registration_and_rng_match_lrsg(self):
        encoder = nn.Sequential(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), nn.Linear(6, 2)).double()
        state = torch.random.get_rng_state().clone()
        actual = GlobalLowRankTransport(encoder, self.config["lrsg"])
        reference = ReferenceTransport(encoder, self.config["lrsg"])
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(set(actual.global_delta_c), set(actual.u))
        self.assertEqual(set(actual.global_delta_c), {"0", "4"})
        for key, parameter in actual.global_delta_c.items():
            self.assertIsInstance(parameter, nn.Parameter)
            self.assertTrue(parameter.requires_grad)
            self.assertEqual(tuple(parameter.shape), (actual.u[key].shape[1],))
            self.assertEqual(parameter.dtype, torch.float64)
            self.assertEqual(int(parameter.count_nonzero()), 0)
        self.assertEqual(actual.architecture()["global_delta_c_layout"], [
            dict(name="0.weight", key="0", rank=4), dict(name="2.weight", key="4", rank=2)])

    def test_zero_coefficients_exact_lrsg_fast_weights_and_query_logits(self):
        # Learned nonzero U/V is essential: U=0 alone could hide a bad coefficient formula.
        for dtype in (torch.float32, torch.float64):
            model = Tiny().to(dtype=dtype)
            weights = list(model.parameters())
            x, query = self.x.to(dtype), self.query.to(dtype)
            reference = ReferenceTransport(model, self.config["lrsg"])
            actual = GlobalLowRankTransport(model, self.config["lrsg"])
            for learned in (False, True):
                with self.subTest(dtype=dtype, learned=learned):
                    if learned:
                        with torch.no_grad():
                            for parameter in reference.parameters():
                                parameter.normal_(0, .3)
                    missing = actual.load_state_dict(reference.state_dict(), strict=False)
                    self.assertEqual(missing.missing_keys, ["global_delta_c.0"])
                    self.assertEqual(missing.unexpected_keys, [])
                    a = helpers.adapt(model, weights, x, self.y, self.cfg, 2, actual)
                    b = reference_helpers.adapt(model, weights, x, self.y, self.cfg, 2, reference)
                    for fast_a, fast_b in zip(a, b, strict=True):
                        torch.testing.assert_close(fast_a, fast_b, rtol=0, atol=0)
                    logits_a, loss_a = helpers.query_loss(model, a, query, self.y)
                    logits_b, loss_b = reference_helpers.query_loss(model, b, query, self.y)
                    torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)
                    shared_a = weights + list(actual.logits.parameters()) + list(actual.u.parameters()) + list(actual.v.parameters())
                    shared_b = weights + list(reference.logits.parameters()) + list(reference.u.parameters()) + list(reference.v.parameters())
                    for grad_a, grad_b in zip(torch.autograd.grad(loss_a, shared_a),
                                              torch.autograd.grad(loss_b, shared_b), strict=True):
                        torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)

    def test_conv_linear_formula_and_bias_bn_scalar_only(self):
        encoder = nn.Sequential(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), nn.Linear(6, 2)).double()
        transport = GlobalLowRankTransport(encoder, dict(self.config["lrsg"], rank=3, beta=.7))
        with torch.no_grad():
            for key in transport.u:
                transport.u[key].normal_(0, .4)
                transport.global_delta_c[key].copy_(torch.linspace(-1.5, .8, len(transport.global_delta_c[key])))
        before = copy.deepcopy(transport.state_dict())
        for name, parameter in encoder.named_parameters():
            key = transport.indices[name]
            gradient = torch.randn_like(parameter, requires_grad=True)
            expected = transport.logits[key].sigmoid() * gradient.detach()
            if key in transport.u:
                matrix = gradient.detach().reshape(gradient.shape[0], -1)
                correction = transport.beta * (transport.u[key] @ torch.diag(1 + transport.global_delta_c[key])
                                               @ transport.v[key].T @ matrix)
                expected = expected + correction.reshape_as(gradient)
            else:
                self.assertEqual(parameter.ndim, 1)
                self.assertNotIn(key, transport.global_delta_c)
            actual = transport.transport_gradient(name, gradient)
            torch.testing.assert_close(actual, expected)
            self.assertIsNone(torch.autograd.grad(actual.sum(), gradient, allow_unused=True)[0])
        for key, value in before.items():
            torch.testing.assert_close(transport.state_dict()[key], value, rtol=0, atol=0)

    def test_outer_loss_gradients_connected_at_init_and_nonzero_after_u_learns(self):
        parameters = self.weights + list(self.transport.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.cfg["outer_lr"])
        helpers.query_loss(self.model, self.adapt(), self.query, self.y)[1].backward()
        coefficient = self.transport.global_delta_c["0"]
        self.assertIsNotNone(coefficient.grad)
        self.assertTrue(torch.isfinite(coefficient.grad).all())
        self.assertEqual(int(coefficient.grad.count_nonzero()), 0)  # U=0
        self.assertGreater(self.transport.u["0"].grad.norm().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        helpers.query_loss(self.model, self.adapt(), self.query, self.y)[1].backward()
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(coefficient.grad.norm().item(), 0)
        before = coefficient.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(coefficient, before))

    def test_inner_loop_keeps_global_parameters_fixed_and_head_ungated(self):
        cfg = dict(self.cfg, encoder_lr=.07, classifier_lr=.13, grad_clip=.025)
        with torch.no_grad():
            self.transport.u["0"].normal_(0, .3)
            self.transport.global_delta_c["0"].normal_(0, .5)
        before = copy.deepcopy(self.transport.state_dict())
        original_grad = torch.autograd.grad
        original_transport = self.transport.transport_gradient
        previous, transported = [], []
        transport_ids = {id(parameter) for parameter in self.transport.parameters()}

        def check_updates(fast):
            old, gradients = previous[-1]
            for current, initial, gradient in zip(fast[-2:], old[-2:], gradients[-2:]):
                torch.testing.assert_close(current, initial - cfg["classifier_lr"] * gradient, rtol=0, atol=0)
            for current, initial, gradient in zip(fast[:-2], old[:-2], transported[-len(self.weights):]):
                torch.testing.assert_close(current, initial - cfg["encoder_lr"] * gradient, rtol=0, atol=0)

        def observe_grad(loss, fast, **kwargs):
            self.assertIs(kwargs["create_graph"], False)
            self.assertIs(kwargs["retain_graph"], True)
            self.assertEqual(len(fast), len(self.weights) + 2)
            self.assertTrue(transport_ids.isdisjoint(id(w) for w in fast))
            if previous:
                check_updates(fast)
            grads = original_grad(loss, fast, **kwargs)
            self.assertTrue(all(g.grad_fn is None and not g.requires_grad for g in grads))
            previous.append(([w.detach().clone() for w in fast],
                             [g.clamp(-cfg["grad_clip"], cfg["grad_clip"]) for g in grads]))
            for key, value in before.items():
                torch.testing.assert_close(self.transport.state_dict()[key], value, rtol=0, atol=0)
            return grads

        def observe_transport(name, gradient):
            self.assertIn(name, dict(self.model.named_parameters()))
            self.assertLessEqual(float(gradient.abs().max()), cfg["grad_clip"])
            result = original_transport(name, gradient)
            transported.append(result.detach().clone())
            return result

        with patch.object(torch.autograd, "grad", side_effect=observe_grad), \
                patch.object(self.transport, "transport_gradient", side_effect=observe_transport):
            fast = self.adapt(cfg=cfg)
        check_updates(fast)
        self.assertEqual(len(previous), cfg["inner_steps"])
        self.assertEqual(len(transported), cfg["inner_steps"] * len(self.weights))
        for key, value in before.items():
            torch.testing.assert_close(self.transport.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.transport.parameters()))

    def test_no_gate_net_condition_or_task_embedding_dependency(self):
        self.assertFalse(hasattr(self.transport, "condition"))
        self.assertFalse(hasattr(self.transport, "gate_net"))
        self.assertFalse(hasattr(self.model, "in_features"))
        self.assertEqual(list(inspect.signature(self.transport.transport_gradient).parameters), ["name", "grad"])
        for path in ROOT.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, ("gate_net", "task_transport"))
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, ("GateNet", "task_embedding", "conditioning"))
        gradient = torch.randn_like(self.weights[0])
        expected = self.transport.transport_gradient("encoder.weight", gradient)
        with patch.object(self.model, "forward_weights", wraps=self.model.forward_weights) as forward:
            self.adapt()
        self.assertEqual(sum(call.kwargs.get("embedding", False) for call in forward.call_args_list), 1)
        # Different support episodes do not change the shared transport of the same G.
        helpers.adapt(self.model, self.weights, self.x + 2, self.y, self.cfg, 2, self.transport)
        torch.testing.assert_close(self.transport.transport_gradient("encoder.weight", gradient), expected, rtol=0, atol=0)

    def test_optimizer_membership_training_checkpoint_and_strict_restore(self):
        config = copy.deepcopy(self.config)
        config["experiment_config"] = dict(train_iterations=4, validation_tasks=1, validate_every=4)
        x = self.x.float()
        task = SimpleNamespace(num_ways=2, support_set=(x, self.y, self.y), query_set=(x + .2, self.y, self.y))
        generator = lambda count: iter([task] * count)
        with patch.object(baseline, "ResNet", Tiny), \
                patch.object(baseline, "read_config", return_value=config), \
                patch.object(torch.cuda, "is_available", return_value=False):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *args, **kw: None))
            optimizer_ids = {id(p) for group in meta.optimizer.param_groups for p in group["params"]}
            self.assertEqual(optimizer_ids, {id(p) for p in meta.weights + list(meta.transport.parameters())})
            self.assertTrue({id(p) for p in meta.transport.global_delta_c.parameters()}.issubset(optimizer_ids))
            self.assertTrue({id(p) for p in meta.weights}.isdisjoint(id(p) for p in meta.transport.parameters()))
            with patch.object(meta.optimizer, "step", wraps=meta.optimizer.step) as step:
                learner = meta.meta_fit(generator, generator)
                self.assertEqual(step.call_count, 2)  # four tasks / unchanged meta-batch of two
            self.assertGreater(int(meta.transport.global_delta_c["0"].count_nonzero()), 0)
            support = (x, self.y, self.y, 2, 3)
            expected = learner.fit(support).predict(x)
            with tempfile.TemporaryDirectory() as directory:
                learner.save(directory)
                restored = baseline.MyLearner()
                restored.load(directory)
                torch.testing.assert_close(torch.from_numpy(expected), torch.from_numpy(restored.fit(support).predict(x)), rtol=0, atol=0)
                for key, value in learner.transport.state_dict().items():
                    torch.testing.assert_close(restored.transport.state_dict()[key], value, rtol=0, atol=0)
                path = Path(directory) / "max-va.pth"
                payload = torch.load(path, weights_only=True)
                self.assertEqual(payload["method"], "fo-proto-global-lrsgmaml")
                self.assertEqual(payload["config"], config)
                self.assertIn("global_delta_c.0", payload["state"]["lrsg"])
                self.assertGreater(int(payload["state"]["lrsg"]["global_delta_c.0"].count_nonzero()), 0)
                wrong = copy.deepcopy(payload)
                wrong["state"]["architecture"]["global_delta_c_layout"][0]["rank"] += 1
                torch.save(wrong, path)
                with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                    baseline.MyLearner().load(directory)
                wrong = copy.deepcopy(payload)
                del wrong["state"]["lrsg"]["global_delta_c.0"]
                torch.save(wrong, path)
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    baseline.MyLearner().load(directory)
                wrong = copy.deepcopy(payload)
                wrong["method"] = "fo-proto-lrsgmaml"
                torch.save(wrong, path)
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    baseline.MyLearner().load(directory)
            best = copy.deepcopy(meta.best_state)
            meta.best_score = 2
            with torch.no_grad():
                meta.transport.global_delta_c["0"].add_(1)
            meta.meta_valid(generator)
            for key, value in best["lrsg"].items():
                torch.testing.assert_close(meta.best_state["lrsg"][key], value, rtol=0, atol=0)

    def test_disabled_transport_and_variable_way_eval_match_existing_protocol(self):
        disabled = GlobalLowRankTransport(self.model, dict(self.config["lrsg"], enabled=False))
        self.assertEqual(list(disabled.parameters()), [])
        expected = reference_helpers.adapt(self.model, self.weights, self.x, self.y, self.cfg, 2)
        for a, b in zip(self.adapt(disabled), expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with torch.no_grad():
            self.transport.u["0"].normal_(0, .2)
            self.transport.global_delta_c["0"].fill_(.5)
        for ways in (2, 7, 20):
            labels = torch.arange(ways).repeat(2)
            support = torch.randn(len(labels), 3, dtype=torch.float64)
            self.transport.train()
            training = helpers.adapt(self.model, self.weights, support, labels, self.cfg, ways, self.transport)
            self.transport.eval()
            before = copy.deepcopy(self.transport.state_dict())
            with torch.no_grad():
                evaluated = helpers.adapt(self.model, self.weights, support, labels, self.cfg, ways, self.transport)
                logits, _ = helpers.query_loss(self.model, evaluated, support, labels)
            self.assertEqual(tuple(logits.shape), (len(labels), ways))
            for a, b in zip(training, evaluated):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key, value in before.items():
                torch.testing.assert_close(self.transport.state_dict()[key], value, rtol=0, atol=0)

    def test_existing_metrics_include_global_coefficient_correction(self):
        with torch.no_grad():
            self.transport.u["0"].fill_(.2)
            self.transport.global_delta_c["0"].copy_(torch.tensor([-1., .5, 2., 3.]))
        gradient = torch.ones_like(self.weights[0])
        self.transport.transport_gradient("encoder.weight", gradient)
        projection = self.transport.v["0"].T @ gradient
        correction = self.transport.beta * (self.transport.u["0"] @ ((1 + self.transport.global_delta_c["0"]).unsqueeze(1) * projection))
        metrics = self.transport.metrics()
        self.assertAlmostEqual(metrics["lrsg/correction_to_gradient_ratio"], float((correction.norm() / (gradient.norm() + 1e-12)).detach()))
        self.transport.reset_metrics()
        self.assertEqual(self.transport._count, 0)

    def test_global_coefficient_metrics_concatenate_layers_and_log_without_mutation(self):
        # Unequal ranks distinguish concatenation from averaging per-layer statistics.
        encoder = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).double()
        transport = GlobalLowRankTransport(encoder, self.config["lrsg"])
        with torch.no_grad():
            transport.global_delta_c["0"].copy_(torch.tensor([-3., 0., 1., 2.]))
            transport.global_delta_c["2"].copy_(torch.tensor([4., -2.]))
        for parameter in transport.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        before = copy.deepcopy(transport.state_dict())
        gradients = [p.grad.clone() for p in transport.parameters()]
        rng = torch.random.get_rng_state().clone()
        values = transport.metrics()
        expected = dict(global_delta_c_mean=2 / 6, global_delta_c_mean_abs=12 / 6,
                        global_delta_c_norm=34 ** .5, global_delta_c_max_abs=4.)
        for key, value in expected.items():
            self.assertIsInstance(values["lrsg/" + key], float)
            self.assertAlmostEqual(values["lrsg/" + key], value)
        calls = []
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(stdout), \
                patch.dict(sys.modules, wandb=SimpleNamespace(run=object(), log=calls.append)):
            log_metrics(SimpleNamespace(logs_dir=directory), transport, 5)
            record = json.loads((Path(directory) / "lrsg_metrics.jsonl").read_text())
        self.assertEqual(record, dict(iteration=5, **values))
        self.assertEqual(calls, [record])
        self.assertEqual(json.loads(stdout.getvalue()), record)
        for key, value in before.items():
            torch.testing.assert_close(transport.state_dict()[key], value, rtol=0, atol=0)
        for parameter, gradient in zip(transport.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_global_coefficient_metrics_zero_empty_and_disabled(self):
        scalar_only = GlobalLowRankTransport(nn.BatchNorm1d(3), self.config["lrsg"])
        self.assertEqual(len(scalar_only.global_delta_c), 0)
        for transport in (self.transport, scalar_only):
            values = transport.metrics()
            for suffix in ("mean", "mean_abs", "norm", "max_abs"):
                self.assertEqual(values["lrsg/global_delta_c_" + suffix], 0.0)
        disabled = GlobalLowRankTransport(self.model, dict(self.config["lrsg"], enabled=False))
        self.assertEqual(disabled.metrics(), {})

    def test_real_resnet_cpu_exact_equivalence_and_outer_backward(self):
        encoder = baseline.make_encoder(dict(num_classes=2, dev=torch.device("cpu"), num_blocks=18, pretrained=False))
        # Default random ResNet prototypes can classify these few support images
        # with exactly zero float32 loss/gradient. Use a synthetic learned BN
        # scale so this fixture exercises a nonzero encoder update in both models.
        with torch.no_grad():
            for name, parameter in encoder.named_parameters():
                if name.endswith(".weight") and parameter.ndim == 1:
                    parameter.fill_(.1)
        weights = list(encoder.parameters())
        actual = GlobalLowRankTransport(encoder, self.config["lrsg"])
        reference = ReferenceTransport(encoder, self.config["lrsg"])
        # Nonzero U makes both the equality test and coefficient-gradient test substantive.
        with torch.no_grad():
            for parameter in reference.u.values():
                parameter.normal_(0, .01)
        actual.load_state_dict(reference.state_dict(), strict=False)
        images = torch.randn(4, 3, 32, 32)
        query_images = torch.randn_like(images)
        labels = torch.tensor([0, 1, 0, 1])
        a = helpers.adapt(encoder, weights, images, labels, self.cfg, 2, actual)
        b = reference_helpers.adapt(encoder, weights, images, labels, self.cfg, 2, reference)
        for fast_a, fast_b in zip(a, b):
            torch.testing.assert_close(fast_a, fast_b, rtol=0, atol=0)
        # Independent queries avoid the zero loss of confidently memorized support images.
        logits_a, loss = helpers.query_loss(encoder, a, query_images, labels)
        logits_b, _ = reference_helpers.query_loss(encoder, b, query_images, labels)
        torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in weights + list(actual.parameters())))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in actual.global_delta_c.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
