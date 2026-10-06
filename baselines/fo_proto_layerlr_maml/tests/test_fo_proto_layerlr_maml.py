"""CPU synthetic checks; no downloaded data or trained checkpoint is required."""
import importlib.util
import io
import json
import math
from contextlib import redirect_stdout
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
import helpers_fo_proto_layerlr_maml as helpers
from layer_lr import LayerwiseInnerLR

# Avoid sharing the generic `model` module with other baseline test suites.
spec = importlib.util.spec_from_file_location("layerlr_submission", ROOT / "model.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
spec = importlib.util.spec_from_file_location(
    "reference_fo_proto_helpers", ROOT.parent / "fo_proto_maml/helpers_fo_proto_maml.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class Tiny(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.model = nn.ModuleDict({"out": nn.Linear(4, 5)})

    def forward_weights(self, x, weights, embedding=False):
        z = F.linear(x, weights[0], weights[1]).tanh()
        return z if embedding else F.linear(z, weights[-2], weights[-1])


class LayerLRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = Tiny().double()
        self.model.model["out"] = nn.Identity()
        self.weights = list(self.model.parameters())
        self.config = baseline.read_config()
        self.cfg = self.config["method_config"]
        self.lrs = baseline.make_inner_lrs(self.model, self.config)
        self.x = torch.randn(6, 3, dtype=torch.double)
        self.y = torch.tensor([1, 0, 1, 1, 0, 1])

    def make(self, per_step=False, steps=5, model=None):
        return LayerwiseInnerLR(self.model if model is None else model,
                                dict(per_step=per_step), self.cfg["encoder_lr"], steps)

    def adapt(self, lrs=None, **overrides):
        return helpers.adapt(self.model, self.weights, self.x, self.y,
                             dict(self.cfg, **overrides), 2, self.lrs if lrs is None else lrs)

    def test_zero_initialization_is_fo_proto_maml_including_meta_gradients(self):
        for dtype, exact in ((torch.double, True), (torch.float, True)):
            model = Tiny().to(dtype)
            model.model["out"] = nn.Identity()
            weights, x = list(model.parameters()), self.x.to(dtype)
            for steps in (1, 5):
                cfg = dict(self.cfg, inner_steps=steps)
                lrs = self.make(steps=steps, model=model)
                expected = reference.adapt(model, weights, x, self.y, cfg, 2)
                actual = helpers.adapt(model, weights, x, self.y, cfg, 2, lrs)
                for a, b in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                query = x + .2
                expected_loss = reference.query_loss(model, expected, query, self.y)[1]
                actual_loss = helpers.query_loss(model, actual, query, self.y)[1]
                for a, b in zip(torch.autograd.grad(actual_loss, weights, retain_graph=True),
                                torch.autograd.grad(expected_loss, weights)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_initialization_consumes_no_rng_and_layout(self):
        rng = torch.get_rng_state().clone()
        shared, stepped = self.make(), self.make(per_step=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(tuple(shared.log_scale.shape), (1, 2))
        self.assertEqual(tuple(stepped.log_scale.shape), (5, 2))
        for lrs in (shared, stepped):
            self.assertEqual(lrs.log_scale.abs().sum().item(), 0.0)
            self.assertEqual(tuple(lrs.rates().shape), (5, 2))
            torch.testing.assert_close(lrs.rates(), torch.full((5, 2), .01, dtype=torch.double),
                                       rtol=0, atol=0)
        self.assertEqual(shared.layout(), dict(
            names=["encoder.weight", "encoder.bias"], shapes=[[4, 3], [4]], inner_steps=5,
            per_step=False, base_lr=.01, parametrization="base_lr * exp(log_scale)"))
        self.assertNotEqual(shared.layout(), stepped.layout())
        with self.assertRaisesRegex(ValueError, "per_step"):
            LayerwiseInnerLR(self.model, dict(per_step=1), .01, 5)
        for bad_lr in (0, -1, float("inf"), float("nan"), "0.01"):
            with self.assertRaisesRegex(ValueError, "encoder_lr"):
                LayerwiseInnerLR(self.model, dict(per_step=False), bad_lr, 5)
        with self.assertRaisesRegex(ValueError, "inner_steps"):
            LayerwiseInnerLR(self.model, dict(per_step=False), .01, 0)
        with_classifier = Tiny().double()
        with self.assertRaisesRegex(ValueError, "classifier removed"):
            LayerwiseInnerLR(with_classifier, dict(per_step=False), .01, 5)

    def test_per_tensor_step_is_unbounded_and_classifier_step_is_unchanged(self):
        lrs = self.make(steps=1)
        with torch.no_grad():
            lrs.log_scale.copy_(torch.tensor([[math.log(250.), math.log(.2)]],
                                             dtype=torch.double))
        cfg = dict(self.cfg, inner_steps=1, grad_clip=.001)
        z = self.model.forward_weights(self.x, self.weights, True)
        initial = [w.clone() for w in self.weights] + list(helpers.prototype_head(z, self.y))
        grads = torch.autograd.grad(
            F.cross_entropy(self.model.forward_weights(self.x, initial), self.y), initial,
            retain_graph=True)
        clipped = [g.detach().clamp(-.001, .001) for g in grads]
        fast = self.adapt(lrs=lrs, inner_steps=1, grad_clip=.001)
        # 250 x and 0.2 x the base step: no upper or lower clamp on the multiplier.
        torch.testing.assert_close(fast[0], initial[0] - 2.5 * clipped[0])
        torch.testing.assert_close(fast[1], initial[1] - .002 * clipped[1])
        for a, w, g in zip(fast[-2:], initial[-2:], clipped[-2:]):
            torch.testing.assert_close(a, w - cfg["classifier_lr"] * g, rtol=0, atol=0)
        torch.testing.assert_close(lrs.rates(), torch.tensor([[2.5, .002]], dtype=torch.double))
        # The stored row is shared by every inner step.
        with torch.no_grad():
            self.lrs.log_scale.copy_(lrs.log_scale)
        torch.testing.assert_close(self.lrs.rates()[:, 0],
                                   torch.full((5,), 2.5, dtype=torch.double))
        self.assertFalse(torch.equal(self.adapt()[0], fast[0]))

    def test_per_step_rows_are_used_in_order(self):
        stepped = self.make(per_step=True, steps=2)
        with torch.no_grad():
            stepped.log_scale.copy_(torch.tensor([[1., 0.], [-1., 2.]], dtype=torch.double))
        cfg = dict(self.cfg, inner_steps=2)
        fast = helpers.adapt(self.model, self.weights, self.x, self.y, cfg, 2, stepped)
        # Replay with explicit rates.
        z = self.model.forward_weights(self.x, self.weights, True)
        manual = [w.clone() for w in self.weights] + list(helpers.prototype_head(z, self.y))
        for row in ((math.e, 1.), (1 / math.e, math.e ** 2)):
            grads = torch.autograd.grad(
                F.cross_entropy(self.model.forward_weights(self.x, manual), self.y), manual,
                retain_graph=True)
            rates = [.01 * row[0], .01 * row[1], .01, .01]
            manual = [w - r * g.detach() for w, r, g in zip(manual, rates, grads)]
        for a, b in zip(fast, manual, strict=True):
            torch.testing.assert_close(a, b)
        with self.assertRaisesRegex(ValueError, "step mismatch"):
            helpers.adapt(self.model, self.weights, self.x, self.y,
                          dict(self.cfg, inner_steps=3), 2, stepped)
        with self.assertRaisesRegex(ValueError, "required"):
            helpers.adapt(self.model, self.weights, self.x, self.y, cfg, 2, None)

    def test_first_order_support_gradients_and_exact_step_size_meta_gradient(self):
        with torch.no_grad():
            self.lrs.log_scale.copy_(torch.tensor([[.7, -.4]], dtype=torch.double))
        original_grad = torch.autograd.grad
        seen = []

        def observe(*args, **kwargs):
            result = original_grad(*args, **kwargs)
            self.assertFalse(kwargs["create_graph"])
            self.assertTrue(all(g.grad_fn is None and not g.requires_grad for g in result))
            seen.append([g.detach().clamp(-self.cfg["grad_clip"], self.cfg["grad_clip"])
                         for g in result])
            return result

        with patch.object(torch.autograd, "grad", side_effect=observe):
            fast = self.adapt()
        self.assertEqual(len(seen), self.cfg["inner_steps"])
        loss = helpers.query_loss(self.model, fast, self.x + .2, self.y)[1]
        actual = original_grad(loss, self.lrs.log_scale, retain_graph=True)[0]
        # First order: theta_K[j] = theta_0[j] - alpha[j] * sum_s g_s[j], g_s constant, so
        # dL/dlog_scale[j] = -alpha[j] * <dL/dtheta_K[j], sum_s g_s[j]>.
        final = original_grad(loss, fast[:2], retain_graph=True)
        rates = self.lrs.rates()[0]
        expected = torch.stack([
            -rates[j] * sum((final[j] * step[j]).sum() for step in seen) for j in range(2)])
        torch.testing.assert_close(actual[0], expected)
        self.assertTrue(torch.isfinite(actual).all() and (actual != 0).all())
        # Detached support gradients leave exactly the encoder identity path.
        direct = original_grad(sum(w.sum() for w in fast[:-2]), self.weights, retain_graph=True)
        for p, g in zip(self.weights, direct):
            torch.testing.assert_close(g, torch.ones_like(p))
        # The head still differentiates through initial support prototypes.
        coeff = [torch.randn_like(w) for w in fast[-2:]]
        through = original_grad(sum((w * c).sum() for w, c in zip(fast[-2:], coeff)), self.weights)
        head = helpers.prototype_head(self.model.forward_weights(self.x, self.weights, True), self.y)
        expected_head = original_grad(sum((w * c).sum() for w, c in zip(head, coeff)), self.weights)
        for a, b in zip(through, expected_head):
            torch.testing.assert_close(a, b)
        # Every per-step row receives its own gradient.
        stepped = self.make(per_step=True)
        fast = self.adapt(lrs=stepped)
        loss = helpers.query_loss(self.model, fast, self.x + .2, self.y)[1]
        per_step = original_grad(loss, stepped.log_scale)[0]
        self.assertEqual(tuple(per_step.shape), (5, 2))
        self.assertTrue((per_step != 0).all())

    def test_variable_way_and_no_episode_state_leakage(self):
        before = baseline.snapshot(self.model, self.lrs)
        first = self.adapt()
        for ways in (2, 7, 20):
            x = torch.randn(ways * 2, 3, dtype=torch.double)
            y = torch.arange(ways).repeat(2)
            with torch.no_grad():
                fast = helpers.adapt(self.model, self.weights, x, y, self.cfg, ways, self.lrs)
                out, _ = helpers.query_loss(self.model, fast, x, y)
            self.assertEqual(out.shape, (ways * 2, ways))
        for a, b in zip(first, self.adapt()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        after = baseline.snapshot(self.model, self.lrs)
        for section in ("encoder", "inner_lr"):
            for key in before[section]:
                torch.testing.assert_close(before[section][key], after[section][key], rtol=0, atol=0)
        self.assertEqual(before["layout"], after["layout"])
        self.assertTrue(all(p.grad is None for p in self.lrs.parameters()))

    def test_protocol_and_copied_backbone(self):
        old = json.loads((ROOT.parent / "fo_proto_maml/config.json").read_text())
        new = baseline.read_config()
        old["method"] = new["method"]
        old["method_config"]["learned_inner_lr"] = True
        old["inner_lr"] = dict(per_step=False)
        self.assertEqual(new, old)
        for name in ("network.py", "api.py", "weight_names.py", "test.py"):
            self.assertEqual((ROOT / name).read_bytes(),
                             (ROOT.parent / "fo_proto_maml" / name).read_bytes())
        for key, value in (("learned_inner_lr", False), ("gradient_transport", True),
                           ("low_rank_transport", True), ("task_conditioned_gate", True),
                           ("first_order", False)):
            broken = json.loads(json.dumps(new))
            broken["method_config"][key] = value
            with self.assertRaises(ValueError):
                baseline.validate_config(broken)
        broken = json.loads(json.dumps(new))
        broken["inner_lr"]["per_step"] = "false"
        with self.assertRaises(ValueError):
            baseline.validate_config(broken)

    def test_metrics_report_multipliers_by_tensor_kind(self):
        with torch.no_grad():
            self.lrs.log_scale.copy_(torch.tensor([[math.log(100.), math.log(.25)]],
                                                  dtype=torch.double))
        values = self.lrs.metrics()
        self.assertEqual(set(values), {
            "inner_lr/scale_min", "inner_lr/scale_geomean", "inner_lr/scale_max",
            "inner_lr/scale_geomean_weight", "inner_lr/scale_geomean_vector"})
        self.assertAlmostEqual(values["inner_lr/scale_min"], .25, places=12)
        self.assertAlmostEqual(values["inner_lr/scale_max"], 100., places=10)
        self.assertAlmostEqual(values["inner_lr/scale_geomean"], 5., places=12)
        self.assertAlmostEqual(values["inner_lr/scale_geomean_weight"], 100., places=10)
        self.assertAlmostEqual(values["inner_lr/scale_geomean_vector"], .25, places=12)
        scales = self.lrs.scales()
        self.assertEqual(list(scales), ["encoder.weight", "encoder.bias"])
        self.assertAlmostEqual(scales["encoder.weight"][0], 100., places=10)
        stepped = self.make(per_step=True, steps=2)
        self.assertIn("inner_lr/scale_geomean_step_2", stepped.metrics())
        self.assertEqual(len(stepped.scales()["encoder.bias"]), 2)

    def test_outer_update_metrics_best_snapshot_and_checkpoint_roundtrip(self):
        cfg = baseline.read_config()
        cfg["experiment_config"] = dict(train_iterations=4, validation_tasks=1, validate_every=2)
        x = self.x.float()
        task = SimpleNamespace(num_ways=2, support_set=(x, self.y, self.y),
                               query_set=(x + .2, self.y, self.y))
        generator = lambda count: iter([task] * count)
        with tempfile.TemporaryDirectory() as logs, \
                patch.object(baseline, "ResNet", Tiny), \
                patch.object(baseline, "read_config", return_value=cfg):
            logger = SimpleNamespace(log=lambda *a, **kw: None, logs_dir=logs)
            meta = baseline.MyMetaLearner(2, 2, logger)
            self.assertEqual(meta.inner_lrs.log_scale.abs().sum().item(), 0.0)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                learner = meta.meta_fit(generator, generator)
            # Adam moved the step sizes together with the encoder.
            self.assertTrue((meta.inner_lrs.log_scale != 0).all())
            self.assertEqual({id(p) for group in meta.optimizer.param_groups for p in group["params"]},
                             {id(p) for p in meta.meta_parameters})
            self.assertEqual(len(meta.optimizer.param_groups), 1)
            self.assertEqual(meta.optimizer.param_groups[0]["lr"], cfg["method_config"]["outer_lr"])
            records = [json.loads(line) for line in
                       (Path(logs) / "inner_lr_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["iteration"] for r in records], [2, 4])
            self.assertEqual(list(records[0]["scales"]), ["encoder.weight", "encoder.bias"])
            self.assertNotEqual(records[0]["scales"], records[1]["scales"])
            printed = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual([p["iteration"] for p in printed], [2, 4])
            self.assertNotIn("scales", printed[0])
            self.assertTrue(all(not p.requires_grad for p in learner.inner_lrs.parameters()))
            support = (x, self.y, self.y, 2, 3)
            expected = learner.fit(support).predict(x)
            frozen = baseline.snapshot(learner.learner, learner.inner_lrs)
            learner.fit((x + 1, self.y, self.y, 2, 3))
            torch.testing.assert_close(torch.tensor(expected),
                                       torch.tensor(learner.fit(support).predict(x)))
            with tempfile.TemporaryDirectory() as directory:
                learner.save(directory)
                restored = baseline.MyLearner()
                restored.load(directory)
                torch.testing.assert_close(torch.tensor(expected),
                                           torch.tensor(restored.fit(support).predict(x)),
                                           rtol=0, atol=0)
                for section in ("encoder", "inner_lr"):
                    for key, value in frozen[section].items():
                        torch.testing.assert_close(value, restored.state[section][key], rtol=0, atol=0)
                path = Path(directory) / "max-va.pth"
                payload = torch.load(path, weights_only=True)
                self.assertEqual(payload["method"], "fo-proto-layerlr-maml")
                self.assertEqual(payload["config"], cfg)
                self.assertEqual(payload["state"]["layout"], meta.inner_lrs.layout())
                # A checkpoint with another layout or method must not load.
                for change in (lambda p: p["state"]["layout"].update(per_step=True),
                               lambda p: p["state"]["layout"].update(base_lr=.02),
                               lambda p: p.update(method="fo-proto-maml")):
                    broken = torch.load(path, weights_only=True)
                    change(broken)
                    torch.save(broken, path)
                    with self.assertRaises(ValueError):
                        baseline.MyLearner().load(directory)
            # A worse validation score keeps the previous best snapshot object.
            best = meta.best_state
            meta.best_score = 2
            with torch.no_grad():
                for p in meta.meta_parameters:
                    p.add_(.1)
            current = baseline.snapshot(meta.meta_learner, meta.inner_lrs)
            meta.meta_valid(generator)
            self.assertIs(meta.best_state, best)
            meta.best_score = -float("inf")
            meta.meta_valid(generator)
            for section in ("encoder", "inner_lr"):
                for key in current[section]:
                    torch.testing.assert_close(current[section][key],
                                               meta.best_state[section][key], rtol=0, atol=0)

    def test_real_encoder_layout_same_initialization_and_query_gradients(self):
        args = dict(num_classes=2, dev=torch.device("cpu"), num_blocks=18, pretrained=False)
        torch.manual_seed(98)
        model = baseline.make_encoder(args)
        lrs = baseline.make_inner_lrs(model, self.config)
        after = torch.get_rng_state().clone()
        torch.manual_seed(98)
        plain = baseline.make_encoder(args)
        # Building the step sizes neither changes nor reorders encoder initialization.
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for a, b in zip(model.parameters(), plain.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(tuple(lrs.log_scale.shape), (1, 60))  # 20 conv + 40 BN affine tensors
        self.assertEqual(lrs.layout()["shapes"][0], [64, 3, 7, 7])
        self.assertEqual(sum(len(shape) >= 2 for shape in lrs.shapes), 20)
        # One image shared by both classes keeps the support loss away from zero.
        x, y = torch.randn(6, 3, 32, 32), torch.tensor([0, 1, 0, 1, 0, 1])
        x[1] = x[0]
        cfg = dict(self.cfg, inner_steps=2)
        lrs = LayerwiseInnerLR(model, dict(per_step=False), cfg["encoder_lr"], 2)
        expected = reference.adapt(model, list(model.parameters()), x, y, cfg, 2)
        fast = helpers.adapt(model, list(model.parameters()), x, y, cfg, 2, lrs)
        for a, b in zip(fast, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        helpers.query_loss(model, fast, x + .1, y)[1].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in list(model.parameters()) + list(lrs.parameters())))
        self.assertGreater(lrs.log_scale.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
