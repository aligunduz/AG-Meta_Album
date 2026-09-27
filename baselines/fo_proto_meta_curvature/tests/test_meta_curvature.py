"""Prepared verification suite; run explicitly with unittest discovery."""
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
import helpers_fo_proto_meta_curvature as helpers
from meta_curvature import EncoderCurvature, ParameterCurvature

# Avoid sharing the generic `model` module with other baseline test suites.
spec = importlib.util.spec_from_file_location("mc2_submission", ROOT / "model.py")
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


class MC2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = Tiny().double()
        self.model.model["out"] = nn.Identity()
        self.weights = list(self.model.parameters())
        self.mc = EncoderCurvature(self.model)
        self.cfg = baseline.read_config()["method_config"]
        self.x = torch.randn(6, 3, dtype=torch.double)
        self.y = torch.tensor([1, 0, 1, 1, 0, 1])

    def adapt(self, **overrides):
        return helpers.adapt(self.model, self.weights, self.x, self.y,
                             dict(self.cfg, **overrides), 2, self.mc)

    def test_fixed_identity_matches_adaptation_and_meta_gradients(self):
        self.mc.requires_grad_(False)
        for steps in (1, 5):
            cfg = dict(self.cfg, inner_steps=steps)
            expected = reference.adapt(self.model, self.weights, self.x, self.y, cfg, 2)
            actual = self.adapt(inner_steps=steps)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
            query = self.x + .2
            expected_loss = reference.query_loss(self.model, expected, query, self.y)[1]
            actual_loss = helpers.query_loss(self.model, actual, query, self.y)[1]
            for a, b in zip(torch.autograd.grad(actual_loss, self.weights),
                            torch.autograd.grad(expected_loss, self.weights)):
                torch.testing.assert_close(a, b)

    def test_nonsymmetric_conv_matches_official_tf_layout_and_index_sum(self):
        g = torch.randn(3, 2, 2, 3, dtype=torch.double)
        mc = ParameterCurvature(g)
        with torch.no_grad():
            for p in mc.parameters():
                p.copy_(torch.randn_like(p))
        # Execute the official reshape/matmul sequence in HWIO layout.
        tf_g = g.permute(2, 3, 1, 0).contiguous()
        tf_out, tf_in, tf_f = mc.Mo.T, mc.Mi, mc.Mf
        temp = tf_g.reshape(-1, 3) @ tf_out
        temp = (tf_f @ temp.reshape(6, -1)).reshape(6, 2, 3)
        temp = tf_in @ temp.permute(1, 0, 2).reshape(2, -1)
        expected = temp.reshape(2, 6, 3).permute(1, 0, 2)
        expected = expected.reshape(2, 3, 2, 3).permute(3, 2, 0, 1)
        torch.testing.assert_close(mc(g), expected)
        # Independent scalar contraction catches transpose/axis errors.
        flat, scalar = g.flatten(2), torch.zeros(3, 2, 6, dtype=g.dtype)
        for o in range(3):
            for i in range(2):
                for f in range(6):
                    scalar[o, i, f] = sum(mc.Mo[o, a] * mc.Mi[i, b]
                        * mc.Mf[f, c] * flat[a, b, c]
                        for a in range(3) for b in range(2) for c in range(6))
        torch.testing.assert_close(mc(g).flatten(2), scalar)

    def test_nonsymmetric_linear_and_unconstrained_vector(self):
        g = torch.randn(4, 3, dtype=torch.double)
        mc = ParameterCurvature(g)
        with torch.no_grad():
            mc.Mo.copy_(torch.randn_like(mc.Mo))
            mc.Mi.copy_(torch.randn_like(mc.Mi))
        torch.testing.assert_close(mc(g), (mc.Mi @ g.T @ mc.Mo.T).T)
        vector = ParameterCurvature(torch.ones(3, dtype=torch.double))
        torch.testing.assert_close(vector.scale, torch.ones_like(vector.scale))
        with torch.no_grad():
            vector.scale.copy_(torch.tensor([-2., 0., 3.]))
        torch.testing.assert_close(vector(torch.tensor([1., 2., 4.], dtype=torch.double)),
                                   torch.tensor([-2., 0., 12.], dtype=torch.double))

    def test_first_order_curvature_query_path_and_initialization_paths(self):
        original_grad = torch.autograd.grad
        seen = []
        def observe(*args, **kwargs):
            result = original_grad(*args, **kwargs)
            self.assertFalse(kwargs["create_graph"])
            self.assertTrue(all(g.grad_fn is None and not g.requires_grad for g in result))
            seen.append(result)
            return result
        with patch.object(torch.autograd, "grad", side_effect=observe):
            fast = self.adapt()
        self.assertEqual(len(seen), self.cfg["inner_steps"])
        loss = helpers.query_loss(self.model, fast, self.x + .2, self.y)[1]
        gradients = original_grad(loss, list(self.mc.parameters()), retain_graph=True)
        self.assertTrue(all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients))
        # Detached support gradients leave exactly the encoder identity path.
        direct = original_grad(sum(w.sum() for w in fast[:-2]), self.weights,
                               retain_graph=True)
        for p, g in zip(self.weights, direct):
            torch.testing.assert_close(g, torch.ones_like(p))
        # The head still differentiates through initial support prototypes.
        coeff = [torch.randn_like(w) for w in fast[-2:]]
        actual = original_grad(sum((w*c).sum() for w, c in zip(fast[-2:], coeff)), self.weights)
        head = helpers.prototype_head(self.model.forward_weights(self.x, self.weights, True), self.y)
        expected = original_grad(sum((w*c).sum() for w, c in zip(head, coeff)), self.weights)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_classifier_update_and_clipping_order(self):
        with torch.no_grad():
            for p in self.mc.parameters():
                p.mul_(3)
        cfg = dict(self.cfg, inner_steps=1, grad_clip=.001)
        z = self.model.forward_weights(self.x, self.weights, True)
        initial = [w.clone() for w in self.weights] + list(helpers.prototype_head(z, self.y))
        grads = torch.autograd.grad(F.cross_entropy(self.model.forward_weights(self.x, initial), self.y),
                                    initial, retain_graph=True)
        clipped = [g.detach().clamp(-.001, .001) for g in grads]
        fast = self.adapt(inner_steps=1, grad_clip=.001)
        for a, w, g in zip(fast[:-2], initial[:-2], self.mc(clipped[:-2])):
            torch.testing.assert_close(a, w - cfg["encoder_lr"] * g)
        for a, w, g in zip(fast[-2:], initial[-2:], clipped[-2:]):
            torch.testing.assert_close(a, w - cfg["classifier_lr"] * g)
        self.assertEqual(self.mc.names, ("encoder.weight", "encoder.bias"))
        with self.assertRaises(ValueError):
            self.mc(clipped)

    def test_variable_way_and_no_episode_state_leakage(self):
        before = baseline.snapshot(self.model, self.mc)
        first = self.adapt()
        for ways in (2, 7, 20):
            x = torch.randn(ways * 2, 3, dtype=torch.double)
            y = torch.arange(ways).repeat(2)
            with torch.no_grad():
                fast = helpers.adapt(self.model, self.weights, x, y, self.cfg, ways, self.mc)
                out, _ = helpers.query_loss(self.model, fast, x, y)
            self.assertEqual(out.shape, (ways * 2, ways))
        for a, b in zip(first, self.adapt()):
            torch.testing.assert_close(a, b)
        after = baseline.snapshot(self.model, self.mc)
        for section in ("encoder", "curvature"):
            for key in before[section]:
                torch.testing.assert_close(before[section][key], after[section][key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.mc.parameters()))

    def test_protocol_and_copied_backbone(self):
        old = json.loads((ROOT.parent / "fo_proto_maml/config.json").read_text())
        new = baseline.read_config()
        old["method"] = new["method"]
        old["method_config"]["gradient_transport"] = True
        self.assertEqual(new, old)
        for name in ("network.py", "api.py", "weight_names.py"):
            self.assertEqual((ROOT / name).read_bytes(), (ROOT.parent / "fo_proto_maml" / name).read_bytes())

    def test_identity_initialization_preserves_random_state(self):
        rng = torch.get_rng_state().clone()
        for shape in ((3, 2, 2, 3), (4, 3), (4,)):
            g = torch.ones(shape, dtype=torch.double)
            mc = ParameterCurvature(g)
            torch.testing.assert_close(mc(g), g, rtol=0, atol=0)
            for name, parameter in mc.named_parameters():
                expected = (torch.ones_like(parameter) if name == "scale"
                            else torch.eye(parameter.shape[0], dtype=parameter.dtype))
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_outer_update_best_snapshot_and_checkpoint_roundtrip(self):
        cfg = baseline.read_config()
        cfg["experiment_config"] = dict(train_iterations=2, validation_tasks=1, validate_every=2)
        x = self.x.float()
        task = SimpleNamespace(num_ways=2, support_set=(x, self.y, self.y),
                               query_set=(x + .2, self.y, self.y))
        generator = lambda count: iter([task] * count)
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=cfg):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))
            initial = [p.detach().clone() for p in meta.curvature.parameters()]
            learner = meta.meta_fit(generator, generator)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial, meta.curvature.parameters())))
            self.assertEqual({id(p) for group in meta.optimizer.param_groups for p in group["params"]},
                             {id(p) for p in meta.meta_parameters})
            self.assertTrue(all(not p.requires_grad for p in learner.curvature.parameters()))
            support = (x, self.y, self.y, 2, 3)
            expected = learner.fit(support).predict(x)
            frozen = baseline.snapshot(learner.learner, learner.curvature)
            learner.fit((x + 1, self.y, self.y, 2, 3))
            torch.testing.assert_close(torch.tensor(expected), torch.tensor(learner.fit(support).predict(x)))
            with tempfile.TemporaryDirectory() as directory:
                learner.save(directory)
                restored = baseline.MyLearner()
                restored.load(directory)
                torch.testing.assert_close(torch.tensor(expected), torch.tensor(restored.fit(support).predict(x)))
                for section in ("encoder", "curvature"):
                    for key, value in frozen[section].items():
                        torch.testing.assert_close(value, restored.state[section][key])
                payload = torch.load(Path(directory) / "max-va.pth", weights_only=True)
                self.assertEqual(payload["method"], "fo-proto-meta-curvature")
                self.assertEqual(payload["config"], cfg)
                self.assertEqual(payload["state"]["architecture"], meta.curvature.architecture())
            best = meta.best_state
            saved_best = {section: {k: v.clone() for k, v in best[section].items()}
                          for section in ("encoder", "curvature")}
            meta.best_score = 2
            with torch.no_grad():
                for p in meta.meta_parameters:
                    p.add_(.1)
            current = baseline.snapshot(meta.meta_learner, meta.curvature)
            meta.meta_valid(generator)
            self.assertIs(meta.best_state, best)
            after = baseline.snapshot(meta.meta_learner, meta.curvature)
            for section in ("encoder", "curvature"):
                for key in current[section]:
                    torch.testing.assert_close(current[section][key], after[section][key])
                    torch.testing.assert_close(saved_best[section][key], meta.best_state[section][key])
            meta.best_score = -float("inf")
            meta.meta_valid(generator)
            for section in ("encoder", "curvature"):
                for key in current[section]:
                    torch.testing.assert_close(current[section][key], meta.best_state[section][key])

    def test_real_encoder_mapping_and_query_gradients(self):
        model = baseline.make_encoder(dict(num_classes=2, dev=torch.device("cpu"),
                                          num_blocks=18, pretrained=False))
        mc = EncoderCurvature(model)
        self.assertEqual(len(mc.architecture()), 60)  # 20 conv + 40 BN affine tensors
        self.assertEqual(mc.architecture()[0]["shape"], [64, 3, 7, 7])
        x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1])
        fast = helpers.adapt(model, list(model.parameters()), x, y,
                             dict(self.cfg, inner_steps=1), 2, mc)
        helpers.query_loss(model, fast, x + .1, y)[1].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in list(model.parameters()) + list(mc.parameters())))


if __name__ == "__main__":
    unittest.main()
