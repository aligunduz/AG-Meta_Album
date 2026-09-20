"""Short CPU checks: python -m unittest discover -s baselines/sgmaml/tests -v."""

import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BASELINES = Path(__file__).resolve().parents[2]


def load_baseline(name):
    """Resolve submission-local imports without polluting other baselines."""
    root = BASELINES / name
    aliases = ("weight_names", "network", "api", "helpers_" + name)
    previous = {alias: sys.modules.get(alias) for alias in aliases}
    try:
        for alias in (*aliases, "model"):
            spec = importlib.util.spec_from_file_location(
                "_test_" + name + "_" + alias, root / (alias + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if alias != "model":
                sys.modules[alias] = module
        return module
    finally:
        for alias, module in previous.items():
            if module is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = module


SGMAML = load_baseline("sgmaml")
MAML = load_baseline("maml")


class TinyNetwork(nn.Module):
    """Nonlinear fast-weight model with the same final weight/bias contract."""

    def __init__(self, num_classes, dev, **kwargs):
        super().__init__()
        self.dev = dev
        self.encoder = nn.Linear(3, 2)
        self.out = nn.Linear(2, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward_weights(self, x, weights):
        features = torch.tanh(F.linear(x, weights[0], weights[1]))
        return F.linear(features, weights[-2], weights[-1])

    def modify_out_layer(self, num_classes):
        self.out = nn.Linear(2, num_classes).to(self.dev)
        nn.init.zeros_(self.out.bias)

    def load_params(self, state):
        self.load_state_dict({k: v for k, v in state.items()
                              if not k.startswith("out.")}, strict=False)


def task(ways=3, shots=2, offset=0.0):
    labels = torch.arange(ways).repeat_interleave(shots)
    x = torch.linspace(-1.1, 1.3, labels.numel() * 3).reshape(-1, 3)
    query_labels = labels.roll(1)
    return SimpleNamespace(
        num_ways=ways, support_set=(x + offset, labels, labels),
        query_set=(x.flip(0) - offset, query_labels, query_labels))


class ScalarGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)
        self.network_patch = mock.patch.object(SGMAML, "ResNet", TinyNetwork)
        self.network_patch.start()
        self.addCleanup(self.network_patch.stop)
        self.device_patch = mock.patch.object(
            SGMAML.MyMetaLearner, "get_device", return_value=torch.device("cpu"))
        self.device_patch.start()
        self.addCleanup(self.device_patch.stop)
        self.logger = SimpleNamespace(log=mock.Mock())

    def make_meta(self):
        return SGMAML.MyMetaLearner(3, 9, self.logger)

    def assert_tensors_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_config_and_training_defaults_match_maml(self):
        original = json.loads((BASELINES / "maml/config.json").read_text())
        gated = json.loads((BASELINES / "sgmaml/config.json").read_text())
        self.assertEqual(gated.pop("gate_init_logit", 4.0), 4.0)
        self.assertEqual(gated, original)
        meta = self.make_meta()
        with mock.patch.object(MAML, "ResNet", TinyNetwork), mock.patch.object(
                MAML.MyMetaLearner, "get_device", return_value=torch.device("cpu")):
            reference = MAML.MyMetaLearner(3, 9, self.logger)
        for name in ("should_train", "ncc", "train_tasks", "val_tasks", "val_after",
                     "base_lr", "grad_clip", "second_order", "meta_batch_size",
                     "T", "lr", "model_args", "opt_fn"):
            self.assertEqual(getattr(meta, name), getattr(reference, name), name)
        self.assertFalse(meta.second_order)

    def test_one_initial_scalar_per_fast_tensor_and_outer_optimizer_membership(self):
        meta = self.make_meta()
        self.assertIsInstance(meta.gate_logits, nn.ParameterList)
        self.assertEqual(len(meta.gate_logits), len(list(meta.meta_learner.parameters())))
        self.assertEqual(len(meta.gate_logits), len(meta.weights))
        for weight, gate in zip(meta.weights, meta.gate_logits):
            self.assertEqual(gate.ndim, 0)
            self.assertEqual(gate.item(), 4.0)
            self.assertTrue(gate.requires_grad)
            self.assertIsNot(weight, gate)
        expected = meta.weights + list(meta.gate_logits)
        self.assertEqual([id(p) for p in meta.meta_parameters], [id(p) for p in expected])
        optimizer_params = [p for group in meta.optimizer.param_groups for p in group["params"]]
        self.assertEqual([id(p) for p in optimizer_params], [id(p) for p in expected])
        self.assertEqual([p.shape for p in meta.grad_buffer], [p.shape for p in expected])

    def test_inner_update_clips_before_distinct_gates_including_resized_head(self):
        weights = [torch.randn(2, 3), torch.randn(7, 2), torch.randn(7)]
        grads = [torch.linspace(-30, 20, p.numel()).reshape_as(p) for p in weights]
        gates = [torch.tensor(value, requires_grad=True) for value in (-4.0, 0.2, 2.0)]
        originals = [p.clone() for p in weights + grads + gates]
        for clip in (None, 1.5):
            with self.subTest(clip=clip):
                result = SGMAML.update_weights(weights, grads, clip, 0.13, gates)
                for w, grad, gate, updated in zip(weights, grads, gates, result):
                    clipped = grad if clip is None else grad.clamp(-clip, clip)
                    torch.testing.assert_close(updated, w - 0.13 * gate.sigmoid() * clipped)
        self.assert_tensors_equal(weights + grads + gates, originals)

    def test_multistep_query_gate_gradients_match_detached_support_reference(self):
        meta = self.make_meta()
        meta.T, meta.base_lr, meta.grad_clip = 3, 0.3, 1.5
        with torch.no_grad():
            for index, gate in enumerate(meta.gate_logits):
                gate.fill_(-0.7 + index * 0.5)
        before = [gate.detach().clone() for gate in meta.gate_logits]
        episode = task()
        xs, ys, _ = episode.support_set
        xq, yq, _ = episode.query_set
        _, loss = meta.compute_out_and_loss(
            meta.meta_learner, [w.clone() for w in meta.weights],
            xs, ys, xq, yq, 3, True)
        self.assert_tensors_equal(meta.gate_logits, before)
        self.assertTrue(all(gate.grad is None for gate in meta.gate_logits))
        observed = torch.autograd.grad(loss, meta.meta_parameters)

        def reference_grads(second_order):
            weights = [w.detach().clone().requires_grad_() for w in meta.weights]
            logits = [a.detach().clone().requires_grad_() for a in meta.gate_logits]
            fast = [w.clone() for w in weights]
            for _ in range(meta.T):
                support_loss = meta.meta_learner.criterion(
                    meta.meta_learner.forward_weights(xs, fast), ys)
                gradients = torch.autograd.grad(
                    support_loss, fast, create_graph=second_order, retain_graph=True)
                if not second_order:
                    self.assertTrue(all(not g.requires_grad for g in gradients))
                fast = [w - meta.base_lr * a.sigmoid() * g.clamp(-1.5, 1.5)
                        for w, a, g in zip(fast, logits, gradients)]
            query_loss = meta.meta_learner.criterion(
                meta.meta_learner.forward_weights(xq, fast), yq)
            return torch.autograd.grad(query_loss, weights + logits)

        expected = reference_grads(False)
        for actual, detached_reference in zip(observed, expected):
            torch.testing.assert_close(actual, detached_reference)
        gates = observed[len(meta.weights):]
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().item() > 0 for g in gates))
        hessian_reference = reference_grads(True)[len(meta.weights):]
        self.assertGreater(max((a - b).abs().item() for a, b in zip(gates, hessian_reference)), 1e-6)

    def test_outer_step_uses_two_tasks_for_weights_and_gates_and_fallback_state(self):
        meta = self.make_meta()
        meta.T, meta.train_tasks, meta.val_after = 2, 2, 99
        meta.optimizer = torch.optim.SGD(meta.meta_parameters, lr=0.2)
        episodes = [task(offset=0.0), task(offset=0.4)]
        before = [p.detach().clone() for p in meta.meta_parameters]
        task_gradients = []
        for episode in episodes:
            xs, ys, _ = episode.support_set
            xq, yq, _ = episode.query_set
            _, loss = meta.compute_out_and_loss(
                meta.meta_learner, [p.clone() for p in meta.weights],
                xs, ys, xq, yq, 3, True)
            task_gradients.append([g.clamp(-meta.grad_clip, meta.grad_clip)
                                   for g in torch.autograd.grad(loss, meta.meta_parameters)])

        def generate(count):
            self.assertEqual(count, 2)
            yield episodes[0]
            self.assert_tensors_equal(meta.meta_parameters, before)
            yield episodes[1]

        with mock.patch.object(meta.optimizer, "step", wraps=meta.optimizer.step) as step:
            learner = meta.meta_fit(generate, mock.Mock())
        self.assertEqual(step.call_count, 1)
        self.assertEqual(self.logger.log.call_count, 2)
        for actual, initial, first, second in zip(meta.meta_parameters, before, *task_gradients):
            torch.testing.assert_close(actual, initial - 0.2 * (first + second))
        self.assertTrue(all(torch.count_nonzero(g) == 0 for g in meta.grad_buffer))
        self.assert_tensors_equal(learner.weights, meta.weights)
        self.assert_tensors_equal(learner.gate_logits, meta.gate_logits)
        self.assertTrue(all(not p.requires_grad for p in learner.gate_logits))

    def test_validation_best_keeps_weights_and_gates_paired_on_ties_and_regressions(self):
        meta = self.make_meta()
        episode = task(ways=2, shots=1)
        labels = episode.query_set[1]
        previous_best = None
        for marker, num_correct in ((1.0, 1), (2.0, 0), (3.0, 1), (4.0, 2)):
            with torch.no_grad():
                for p in meta.weights:
                    p.fill_(marker)
                for gate in meta.gate_logits:
                    gate.fill_(-marker)
            predictions = labels.clone()
            predictions[num_correct:] = 1 - predictions[num_correct:]
            logits = F.one_hot(predictions, num_classes=2).float()
            with mock.patch.object(meta, "compute_out_and_loss", return_value=(logits, None)):
                meta.meta_valid(lambda _: [episode])
            if marker in (1.0, 4.0):
                previous_best = ([p.detach().clone() for p in meta.weights],
                                 [p.detach().clone() for p in meta.gate_logits])
            self.assert_tensors_equal(meta.best_state, previous_best[0])
            self.assert_tensors_equal(meta.best_gate_logits, previous_best[1])
        with torch.no_grad():
            meta.weights[0].add_(10)
            meta.gate_logits[0].add_(10)
        meta.should_train = False
        learner = meta.meta_fit(mock.Mock(), mock.Mock())
        self.assert_tensors_equal(learner.weights, previous_best[0])
        self.assert_tensors_equal(learner.gate_logits, previous_best[1])
        self.assertTrue(all(not p.requires_grad for p in meta.best_state + meta.best_gate_logits))

    def saved_learner(self, directory):
        meta = self.make_meta()
        meta.should_train, meta.T = False, 2
        with torch.no_grad():
            for index, gate in enumerate(meta.gate_logits):
                gate.fill_(-2.0 + index)
        source = meta.meta_fit(mock.Mock(), mock.Mock())
        source.save(directory)
        loaded = SGMAML.MyLearner()
        loaded.load(directory)
        return source, loaded

    def test_checkpoint_roundtrip_and_any_way_any_shot_keep_learned_gates_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, loaded = self.saved_learner(directory)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {
                "model_args.pickle", "model_state.pickle", "weights.pickle",
                "maml_params.pickle", "gate_logits.pickle"})
            self.assert_tensors_equal(loaded.weights, source.weights)
            self.assert_tensors_equal(loaded.gate_logits, source.gate_logits)
            self.assertTrue(all(not p.requires_grad for p in loaded.gate_logits))
            weights_before = [p.detach().clone() for p in loaded.weights]
            gates_before = [p.clone() for p in loaded.gate_logits]
            with mock.patch.object(SGMAML, "update_weights", wraps=SGMAML.update_weights) as update:
                for ways, shots in ((2, 1), (5, 2)):
                    with self.subTest(ways=ways, shots=shots):
                        episode = task(ways, shots)
                        predictor = loaded.fit((*episode.support_set, ways, shots))
                        probabilities = predictor.predict(episode.query_set[0])
                        self.assertEqual(probabilities.shape, (ways * shots, ways))
                        self.assertTrue(np.isfinite(probabilities).all())
                        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
                        self.assertEqual(predictor.weights[-2].shape, (ways, 2))
                        self.assertEqual(predictor.weights[-1].shape, (ways,))
            self.assertEqual(update.call_count, 2 * loaded.T)
            for call in update.call_args_list:
                self.assertIs(call.args[-1], loaded.gate_logits)
            self.assert_tensors_equal(loaded.gate_logits, gates_before)
            self.assert_tensors_equal(loaded.weights, weights_before)

    def test_checkpoint_rejects_missing_mismatched_or_nonscalar_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.saved_learner(directory)
            gate_file = Path(directory) / "gate_logits.pickle"
            for gates in (source.gate_logits[:-1],
                          [torch.ones(1)] + source.gate_logits[1:]):
                with gate_file.open("wb") as handle:
                    pickle.dump(gates, handle)
                with self.assertRaises(ValueError):
                    SGMAML.MyLearner().load(directory)
            gate_file.unlink()
            with self.assertRaisesRegex(Exception, "gate_logits"):
                SGMAML.MyLearner().load(directory)


if __name__ == "__main__":
    unittest.main()
