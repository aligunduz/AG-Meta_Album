"""Unit tests authored for later execution; no dataset or frozen encoder needed."""
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


def load_submission():
    aliases = ("weight_names", "network", "api", "task_gates",
               "helpers_fo_proto_tcsgmaml", "model")
    previous = {name: sys.modules.get(name) for name in aliases}
    loaded = {}
    try:
        for name in aliases:
            spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[name] = sys.modules[name] = module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return loaded


modules = load_submission()
baseline = modules["model"]
helpers = modules["helpers_fo_proto_tcsgmaml"]
TaskGates = modules["task_gates"].TaskGates


class Tiny(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.in_features = 4
        self.model = nn.ModuleDict({"out": nn.Linear(4, 5)})

    def forward_weights(self, x, weights, embedding=False):
        features = F.linear(x, weights[0], weights[1]).tanh()
        return features if embedding else F.linear(features, weights[-2], weights[-1])


class SharedEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        cpu_only = patch.object(torch.cuda, "is_available", return_value=False)
        cpu_only.start()
        self.addCleanup(cpu_only.stop)
        torch.manual_seed(18)
        self.model = Tiny()
        self.model.model["out"] = nn.Identity()
        self.weights = list(self.model.parameters())
        self.config = baseline.read_config()
        self.cfg = self.config["method_config"]
        self.gates = TaskGates(self.model, self.config)
        self.support = torch.randn(7, 3)
        self.labels = torch.tensor([1, 0, 1, 1, 0, 1, 0])
        self.query = torch.randn(4, 3)
        self.targets = torch.tensor([0, 1, 1, 0])

    def adapt(self, **overrides):
        return helpers.adapt(self.model, self.weights, self.support, self.labels,
                             dict(self.cfg, **overrides), self.gates)

    def test_same_main_encoder_feature_tensor_in_both_branches(self):
        captured = {}
        forward = self.model.forward_weights
        prototype = helpers.prototype_head
        def capture_forward(x, weights, embedding=False):
            self.assertIs(x, self.support)
            self.assertIs(weights, self.weights)
            result = forward(x, weights, embedding)
            captured["features"] = result
            return result
        def capture_prototype(features, labels, num_classes=None):
            self.assertIs(features, captured["features"])
            return prototype(features, labels, num_classes)
        hook = self.gates.gate_net.register_forward_pre_hook(
            lambda module, inputs: captured.update(embedding=inputs[0]))
        try:
            with patch.object(self.model, "forward_weights", side_effect=capture_forward) as call:
                with patch.object(helpers, "prototype_head", side_effect=capture_prototype):
                    (w, b), gates = helpers.initialize_task(
                        self.model, self.weights, self.support, self.labels, self.gates)
            self.assertEqual(call.call_count, 1)
        finally:
            hook.remove()
        features = captured["features"]
        torch.testing.assert_close(captured["embedding"], features.mean(0).detach())
        self.assertFalse(captured["embedding"].requires_grad)
        self.assertIsNone(captured["embedding"].grad_fn)
        centers = torch.stack([features[self.labels == k].mean(0) for k in range(2)])
        torch.testing.assert_close(w, 2 * centers)
        torch.testing.assert_close(b, -centers.square().sum(1))
        q = torch.randn(5, 4)  # No query encoder branch can mask this check.
        torch.testing.assert_close(F.linear(q, w, b).softmax(1),
                                   (-torch.cdist(q, centers).square()).softmax(1))
        gradients = torch.autograd.grad(F.linear(q, w, b).square().mean(), self.weights)
        self.assertGreater(gradients[0].norm().item(), 0)
        self.assertTrue(gates.requires_grad)

    def test_conditioning_branch_cannot_backpropagate_to_encoder(self):
        # Nonzero output weights avoid a zero-initialization masking false pass.
        with torch.no_grad():
            self.gates.gate_net.out.weight.fill_(0.2)
        _, gates = helpers.initialize_task(self.model, self.weights, self.support,
                                           self.labels, self.gates)
        grads = torch.autograd.grad(gates.sum(), self.weights, allow_unused=True)
        self.assertTrue(all(g is None for g in grads))

    def test_first_order_outer_gradients_reach_gate_net_and_shared_logits(self):
        original = torch.autograd.grad
        inner_calls = []
        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            self.assertFalse(kwargs["create_graph"])
            self.assertTrue(all(g.grad_fn is None and not g.requires_grad for g in result))
            self.assertEqual(len(args[1]), len(self.weights) + 2)
            inner_calls.append(result)
            return result
        with patch.object(torch.autograd, "grad", side_effect=record):
            fast = self.adapt()
        self.assertEqual(len(inner_calls), self.cfg["inner_steps"])
        _, loss = helpers.query_loss(self.model, fast, self.query, self.targets)
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in self.gates.shared_logits), 0)
        self.assertGreater(self.gates.gate_net.out.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.weights[0].grad.norm().item(), 0)
        # TCSGMAML zero output init makes hidden-layer gradients zero initially.
        torch.testing.assert_close(self.gates.gate_net.hidden.weight.grad,
                                   torch.zeros_like(self.gates.gate_net.hidden.weight))
        self.gates.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.gates.gate_net.out.weight.normal_(std=.1)
        fast = self.adapt()
        helpers.query_loss(self.model, fast, self.query, self.targets)[1].backward()
        self.assertGreater(self.gates.gate_net.hidden.weight.grad.abs().sum().item(), 0)

    def test_prototype_jacobian_survives_multiple_inner_steps(self):
        fast = self.adapt()
        a, b = torch.randn_like(fast[-2]), torch.randn_like(fast[-1])
        actual = torch.autograd.grad((a * fast[-2]).sum() + (b * fast[-1]).sum(), self.weights)
        head, _ = helpers.initialize_task(self.model, self.weights, self.support,
                                          self.labels, self.gates)
        expected = torch.autograd.grad((a * head[0]).sum() + (b * head[1]).sum(), self.weights)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)
        self.assertGreater(actual[0].norm().item(), 0)

    def test_exact_update_formula_clip_before_gate_and_ungated_head(self):
        fast = [torch.tensor([2., -1.]), torch.tensor([3.]),
                torch.ones(7, 2), torch.ones(7)]
        grads = [torch.tensor([20., -30.]), torch.tensor([4.]),
                 torch.full((7, 2), 12.), torch.full((7,), -15.)]
        gates = torch.tensor([.25, .75], requires_grad=True)
        config = dict(self.cfg, grad_clip=5, encoder_lr=.2, classifier_lr=.3)
        updated = helpers.update_weights(fast, grads, gates, config)
        expected = [torch.tensor([1.75, -.75]), torch.tensor([2.4]),
                    torch.full((7, 2), -.5), torch.full((7,), 2.5)]
        for left, right in zip(updated, expected):
            torch.testing.assert_close(left, right)
        self.assertTrue(updated[0].requires_grad)
        self.assertFalse(updated[-1].requires_grad)
        with self.assertRaises(ValueError):
            helpers.update_weights(fast, grads, torch.ones(4), config)

    def test_inner_coordinates_hold_initial_head_fixed(self):
        features = self.model.forward_weights(self.support, self.weights, True)
        w, b = helpers.prototype_head(features, self.labels)
        fixed_head_loss = F.cross_entropy(F.linear(features, w.detach(), b.detach()), self.labels)
        expected = torch.autograd.grad(fixed_head_loss, self.weights)
        fast = self.adapt(inner_steps=1, grad_clip=None)
        gate = torch.sigmoid(torch.tensor(self.config["gate_init_logit"]))
        for w, f, g in zip(self.weights, fast, expected):
            torch.testing.assert_close(f, w - self.cfg["encoder_lr"] * gate * g)

    def test_gates_once_per_episode_reused_and_no_inner_gate_update(self):
        original = helpers.update_weights
        seen = []
        before = baseline.cpu_state(self.gates)
        def capture(fast, grads, gates, config):
            seen.append(gates)
            return original(fast, grads, gates, config)
        with patch.object(self.gates.gate_net, "forward", wraps=self.gates.gate_net.forward) as call:
            with patch.object(helpers, "update_weights", side_effect=capture):
                self.adapt()
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(seen), self.cfg["inner_steps"])
        self.assertTrue(all(g is seen[0] for g in seen))
        for key, value in before.items():
            torch.testing.assert_close(value, self.gates.state_dict()[key])

    def test_any_way_and_no_grad_validation(self):
        count = len(self.weights)
        self.assertEqual(len(self.gates.shared_logits), count)
        self.assertEqual(self.gates.gate_net.num_gates, count)
        for ways in (2, 7, 20):
            labels = torch.arange(ways).repeat(2).flip(0)
            support = torch.randn(len(labels), 3)
            with torch.no_grad():
                fast = helpers.adapt(self.model, self.weights, support, labels,
                                     self.cfg, self.gates, ways)
                out, _ = helpers.query_loss(self.model, fast, support, labels)
            self.assertEqual(out.shape, (len(labels), ways))
            self.assertEqual(fast[-2].shape, (ways, 4))
            self.assertEqual(self.gates.gate_net.num_gates, count)

    def test_support_only_and_episode_independence(self):
        query = self.query.clone().requires_grad_()
        head, gates = helpers.initialize_task(self.model, self.weights, self.support,
                                              self.labels, self.gates)
        grad = torch.autograd.grad(head[0].sum() + gates.sum(), query, allow_unused=True)[0]
        self.assertIsNone(grad)
        before = baseline.cpu_state(self.model)
        first = self.adapt()
        helpers.adapt(self.model, self.weights, self.support + 2, self.labels,
                      self.cfg, self.gates)
        again = self.adapt()
        for left, right in zip(first, again):
            torch.testing.assert_close(left, right)
        for name, value in before.items():
            torch.testing.assert_close(value, self.model.state_dict()[name])

    def test_config_and_gate_initialization_parity(self):
        reference = json.loads((ROOT.parent / "fo_proto_maml/config.json").read_text())
        reference["method"] = "fo-proto-tcsgmaml"
        reference["method_config"]["task_conditioned_gate"] = True
        tcsg = json.loads((ROOT.parent / "tcsgmaml/config.json").read_text())
        reference.update(gate_init_logit=tcsg["gate_init_logit"], gate_net=tcsg["gate_net"])
        self.assertEqual(self.config, reference)
        self.assertNotIn("task_encoder", self.config)
        state = torch.random.get_rng_state().clone()
        gates = TaskGates(self.model, self.config)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(all(p.ndim == 0 and p.item() == 4 for p in gates.shared_logits))
        self.assertEqual(gates.gate_net.hidden.out_features, 128)
        self.assertEqual(gates.gate_net.input_norm, "none")
        torch.testing.assert_close(gates.gate_net.out.weight, torch.zeros_like(gates.gate_net.out.weight))
        torch.testing.assert_close(gates.gate_net.out.bias, torch.zeros_like(gates.gate_net.out.bias))
        embedding = torch.randn(4)
        torch.testing.assert_close(gates(embedding), torch.sigmoid(torch.full((2,), 4.)))
        gates.delta_scale = .5
        with torch.no_grad():
            gates.gate_net.out.bias.fill_(2)
        torch.testing.assert_close(gates(embedding), torch.sigmoid(torch.full((2,), 5.)))

    def make_meta(self):
        config = baseline.read_config()
        config["experiment_config"] = dict(train_iterations=2, validation_tasks=1, validate_every=2)
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=config):
            return baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))

    def episode(self, offset=0.):
        return SimpleNamespace(num_ways=2,
                               support_set=(self.support + offset, self.labels, self.labels),
                               query_set=(self.query, self.targets, self.targets))

    def test_meta_batch_accumulates_all_three_groups_before_step(self):
        meta = self.make_meta()
        expected_ids = [id(p) for p in (meta.weights + list(meta.task_gates.parameters()))]
        actual_ids = [id(p) for group in meta.optimizer.param_groups for p in group["params"]]
        self.assertEqual(actual_ids, expected_ids)
        self.assertEqual(len(actual_ids), len(set(actual_ids)))
        episodes = [self.episode(), self.episode(.2)]
        expected = [torch.zeros_like(p) for p in meta.meta_parameters]
        for task in episodes:
            fast = helpers.adapt(meta.meta_learner, meta.weights, task.support_set[0],
                                 self.labels, meta.params, meta.task_gates, 2)
            loss = helpers.query_loss(meta.meta_learner, fast, self.query, self.targets)[1]
            grads = torch.autograd.grad(loss, meta.meta_parameters)
            for total, grad in zip(expected, grads):
                total.add_(grad.clamp(-meta.params["grad_clip"], meta.params["grad_clip"]))
        step = meta.optimizer.step
        captured = []
        def checked_step():
            captured.append([p.grad.clone() for p in meta.meta_parameters])
            return step()
        with patch.object(baseline, "ResNet", Tiny), patch.object(meta.optimizer, "step", side_effect=checked_step):
            meta.meta_fit(lambda count: iter(episodes), lambda count: iter(episodes[:1]))
        self.assertEqual(len(captured), 1)
        for actual, wanted in zip(captured[0], expected):
            torch.testing.assert_close(actual, wanted)
        gate_start = len(meta.weights)
        self.assertGreater(sum(g.abs().sum().item() for g in captured[0][gate_start:]), 0)

    def test_best_snapshot_roundtrip_all_components_without_task_head(self):
        meta = self.make_meta()
        generator = lambda count: iter([self.episode()] * count)
        with patch.object(baseline, "ResNet", Tiny):
            learner = meta.meta_fit(generator, generator)
            self.assertEqual(set(meta.best_state), {"encoder", "task_gates", "gate_architecture"})
            self.assertFalse(any(name.startswith("model.out") for name in meta.best_state["encoder"]))
            self.assertTrue(any(name.startswith("shared_logits.") for name in meta.best_state["task_gates"]))
            self.assertTrue(any(name.startswith("gate_net.") for name in meta.best_state["task_gates"]))
            best = meta.best_state
            reference = {name: {k: v.clone() for k, v in best[name].items()}
                         for name in ("encoder", "task_gates")}
            meta.best_score = 2
            with torch.no_grad():
                for p in meta.meta_parameters:
                    p.add_(.1)
            meta.meta_valid(generator)
            self.assertIs(meta.best_state, best)
            for group in reference:
                for key in reference[group]:
                    torch.testing.assert_close(best[group][key], reference[group][key])
            support = (self.support, self.labels, self.labels, 2, 3)
            prediction = learner.fit(support).predict(self.query)
            with tempfile.TemporaryDirectory() as directory:
                learner.save(directory)
                checkpoint = Path(directory) / "max-va.pth"
                data = torch.load(checkpoint, weights_only=True)
                self.assertEqual(data["method"], "fo-proto-tcsgmaml")
                self.assertEqual(data["config"], learner.config)
                restored = baseline.MyLearner()
                restored.load(checkpoint)
                torch.testing.assert_close(torch.tensor(prediction),
                                           torch.tensor(restored.fit(support).predict(self.query)))
                for group in reference:
                    for key in reference[group]:
                        torch.testing.assert_close(restored.state[group][key], reference[group][key])
                restored.load(directory)
                with self.assertRaises(ValueError):
                    restored.load(Path(directory) / "epoch-last.pth")
                data["state"]["gate_architecture"]["gate_net"]["num_gates"] += 1
                torch.save(data, checkpoint)
                with self.assertRaises(ValueError):
                    restored.load(checkpoint)


if __name__ == "__main__":
    unittest.main()
