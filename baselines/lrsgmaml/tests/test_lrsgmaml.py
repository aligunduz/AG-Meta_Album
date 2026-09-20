"""Focused LRSGMAML checks, written but not executed during implementation.

Optional later execution: python -m unittest discover -s baselines/lrsgmaml/tests -v
"""

import copy
import importlib.util
import json
import pickle
import random
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
    """Resolve submission-local imports without leaking aliases to other baselines."""
    root = BASELINES / name
    aliases = ["weight_names", "network", "api"]
    if name == "lrsgmaml":
        aliases.append("low_rank_transport")
    aliases.extend(["helpers_" + name, "model"])
    previous = {}
    try:
        for alias in aliases:
            unique_name = "_test_" + name + "_" + alias
            spec = importlib.util.spec_from_file_location(unique_name, root / (alias + ".py"))
            module = importlib.util.module_from_spec(spec)
            for key in (unique_name, alias):
                previous[key] = sys.modules.get(key)
                sys.modules[key] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for key, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous_module


SGMAML = load_baseline("sgmaml")
LRSGMAML = load_baseline("lrsgmaml")


class TinyConvNetwork(nn.Module):
    """A nonlinear Conv2d backbone and variable-way final weight/bias pair."""

    def __init__(self, num_classes, dev, **kwargs):
        super().__init__()
        self.dev = dev
        self.conv = nn.Conv2d(2, 3, kernel_size=1, bias=True)
        self.out = nn.Linear(3, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward_weights(self, x, weights):
        features = torch.tanh(F.conv2d(x, weights[0], weights[1])).mean(dim=(2, 3))
        return F.linear(features, weights[-2], weights[-1])

    def modify_out_layer(self, num_classes):
        self.out = nn.Linear(3, num_classes).to(self.dev)
        nn.init.zeros_(self.out.bias)

    def load_params(self, state):
        self.load_state_dict({key: value for key, value in state.items()
                              if not key.startswith("out.")}, strict=False)


def task(ways=3, shots=2, offset=0.0):
    labels = torch.arange(ways).repeat_interleave(shots)
    values = torch.linspace(-1.1, 1.3, labels.numel() * 8).reshape(-1, 2, 2, 2)
    query_labels = labels.roll(1)
    return SimpleNamespace(
        num_ways=ways,
        support_set=(values + offset, labels, labels),
        query_set=(values.flip(0) - offset, query_labels, query_labels))


class LowRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)
        for baseline in (SGMAML, LRSGMAML):
            network_patch = mock.patch.object(baseline, "ResNet", TinyConvNetwork)
            network_patch.start()
            self.addCleanup(network_patch.stop)
            device_patch = mock.patch.object(
                baseline.MyMetaLearner, "get_device", return_value=torch.device("cpu"))
            device_patch.start()
            self.addCleanup(device_patch.stop)
        self.logger = SimpleNamespace(log=mock.Mock())

    def make_meta(self):
        return LRSGMAML.MyMetaLearner(3, 9, self.logger)

    def assert_tensors_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for result, reference in zip(actual, expected):
                self.assert_state_equal(result, reference)
        else:
            self.assertEqual(actual, expected)

    def query_loss(self, meta):
        episode = task()
        xs, ys, _ = episode.support_set
        xq, yq, _ = episode.query_set
        return meta.compute_out_and_loss(
            meta.meta_learner, [p.clone() for p in meta.weights],
            xs, ys, xq, yq, episode.num_ways, True)[1]

    def test_defaults_rng_parity_and_outer_parameter_membership(self):
        original = json.loads((BASELINES / "sgmaml/config.json").read_text())
        updated = json.loads((BASELINES / "lrsgmaml/config.json").read_text())
        self.assertEqual(updated.pop("low_rank"), {"rank": 4})
        self.assertEqual(updated, original)

        before = torch.get_rng_state().clone()
        meta = self.make_meta()
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        reference = SGMAML.MyMetaLearner(3, 9, self.logger)
        self.assert_tensors_equal(meta.weights, reference.weights)
        self.assert_tensors_equal(list(meta.val_learner.parameters()),
                                  list(reference.val_learner.parameters()))
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for name in ("train_tasks", "val_tasks", "val_after", "base_lr", "grad_clip",
                     "second_order", "meta_batch_size", "T", "lr", "model_args", "opt_fn"):
            self.assertEqual(getattr(meta, name), getattr(reference, name), name)
        self.assertFalse(meta.second_order)
        self.assertTrue(all(g.ndim == 0 and g.item() == 4.0 for g in meta.gate_logits))
        expected = meta.weights + list(meta.gate_logits) + list(meta.low_rank_transport.parameters())
        self.assertEqual([id(p) for p in meta.meta_parameters], [id(p) for p in expected])
        optimized = [p for group in meta.optimizer.param_groups for p in group["params"]]
        self.assertEqual([id(p) for p in optimized], [id(p) for p in expected])
        self.assertEqual([g.shape for g in meta.grad_buffer], [p.shape for p in expected])

    def test_only_actual_convolution_weights_get_factors_and_rank_is_validated(self):
        network = nn.Module()
        network.register_parameter("nonconv_4d", nn.Parameter(torch.ones(2, 2, 1, 1)))
        network.stem = nn.Conv2d(2, 3, 1, bias=True)
        network.bn = nn.BatchNorm2d(3)
        network.residual = nn.ModuleDict({
            "conv": nn.Conv2d(3, 5, 3, padding=1, bias=False),
            "projection": nn.Conv2d(3, 5, 1, bias=False)})
        network.out = nn.Linear(5, 2)
        transport = LRSGMAML.LowRankTransport(network, rank=4)
        names = tuple(name for name, _ in network.named_parameters())
        self.assertEqual(transport.weight_names, names)
        self.assertEqual(set(transport.conv_names), {
            "stem.weight", "residual.conv.weight", "residual.projection.weight"})
        for name, parameter in network.named_parameters():
            corrected = transport.correction(name, torch.ones_like(parameter))
            if name in transport.conv_names:
                key = transport.name_to_key[name]
                expected_shape = (parameter.shape[0], min(4, parameter.shape[0]))
                self.assertEqual(tuple(transport.U[key].shape), expected_shape)
                self.assertEqual(tuple(transport.V[key].shape), expected_shape)
                self.assertEqual(corrected.shape, parameter.shape)
                self.assertEqual(torch.count_nonzero(corrected).item(), 0)
                self.assertGreater(torch.count_nonzero(transport.V[key]).item(), 0)
            else:
                self.assertIsNone(corrected)
        expected_count = 2 * (3 * 3 + 5 * 4 + 5 * 4)
        self.assertEqual(sum(p.numel() for p in transport.parameters()), expected_count)
        for invalid in (0, -1, True, 1.5, "4"):
            with self.subTest(rank=invalid), self.assertRaises((TypeError, ValueError)):
                LRSGMAML.LowRankTransport(network, rank=invalid)
        with self.assertRaises(ValueError):
            transport.validate_fast_weights(tuple(reversed(names)), list(network.parameters()))

    def test_zero_u_matches_sgmaml_including_clipping_and_resized_classifier(self):
        network = TinyConvNetwork(3, torch.device("cpu"))
        transport = LRSGMAML.LowRankTransport(network, rank=4)
        for ways in (3, 7):
            network.modify_out_layer(ways)
            weights = list(network.parameters())
            names = tuple(name for name, _ in network.named_parameters())
            gradients = [torch.linspace(-30, 20, p.numel()).reshape_as(p) for p in weights]
            gates = [torch.tensor(-1.0 + i * 0.7) for i in range(len(weights))]
            before = [p.detach().clone() for p in weights + gradients + gates]
            for clip in (None, 1.5):
                with self.subTest(ways=ways, clip=clip):
                    expected = SGMAML.update_weights(weights, gradients, clip, 0.13, gates)
                    observed = LRSGMAML.update_weights(
                        weights, gradients, clip, 0.13, gates, names, transport)
                    self.assert_tensors_equal(observed, expected)
            self.assert_tensors_equal(weights + gradients + gates, before)

    def test_nonzero_low_rank_term_uses_clipped_gradient_without_extra_gate(self):
        network = TinyConvNetwork(3, torch.device("cpu"))
        transport = LRSGMAML.LowRankTransport(network, rank=2)
        with torch.no_grad():
            for u in transport.U.values():
                u.fill_(0.25)
            for v in transport.V.values():
                v.fill_(0.4)
        weights = list(network.parameters())
        gradients = [torch.linspace(-30, 20, p.numel()).reshape_as(p) for p in weights]
        gates = [torch.tensor(-2.0) for _ in weights]
        result = LRSGMAML.update_weights(
            weights, gradients, 1.5, 0.13, gates, transport.weight_names, transport)
        for index, (name, weight, grad, gate) in enumerate(
                zip(transport.weight_names, weights, gradients, gates)):
            clipped = grad.clamp(-1.5, 1.5)
            expected = weight - 0.13 * gate.sigmoid() * clipped
            if name in transport.conv_names:
                key = transport.name_to_key[name]
                g = clipped.reshape(weight.shape[0], -1)
                correction = transport.U[key] @ (transport.V[key].T @ g)
                expected = expected - 0.13 * correction.reshape_as(weight)
            torch.testing.assert_close(result[index], expected)

    def test_multistep_first_order_query_learns_gate_u_and_then_v(self):
        meta = self.make_meta()
        meta.T, meta.base_lr, meta.grad_clip = 3, 0.3, 1.5
        before = [p.detach().clone() for p in meta.meta_parameters]
        recorded_support = []
        original_get_grads = LRSGMAML.get_grads

        def record_support(*args, **kwargs):
            gradients = original_get_grads(*args, **kwargs)
            recorded_support.extend(gradients)
            return gradients

        with mock.patch.object(LRSGMAML, "get_grads", side_effect=record_support):
            loss = self.query_loss(meta)
        self.assertTrue(recorded_support)
        self.assertTrue(all(not grad.requires_grad for grad in recorded_support))
        self.assert_tensors_equal(meta.meta_parameters, before)
        self.assertTrue(all(p.grad is None for p in meta.meta_parameters))
        gradients = torch.autograd.grad(loss, meta.meta_parameters)
        by_parameter = {id(parameter): gradient for parameter, gradient
                        in zip(meta.meta_parameters, gradients)}
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        for group in (meta.weights, meta.gate_logits, meta.low_rank_transport.U.values()):
            self.assertGreater(sum(by_parameter[id(p)].abs().sum().item() for p in group), 0)
        self.assertTrue(all(torch.count_nonzero(by_parameter[id(v)]).item() == 0
                            for v in meta.low_rank_transport.V.values()))
        with torch.no_grad():
            for u in meta.low_rank_transport.U.values():
                u.add_(0.1)
        v_gradients = torch.autograd.grad(
            self.query_loss(meta), list(meta.low_rank_transport.V.values()))
        self.assertTrue(all(torch.isfinite(g).all() for g in v_gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in v_gradients), 0)

    def test_outer_step_accumulates_two_tasks_for_factors_as_well_as_gates(self):
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
                xs, ys, xq, yq, episode.num_ways, True)
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
        for actual, initial, first, second in zip(meta.meta_parameters, before, *task_gradients):
            torch.testing.assert_close(actual, initial - 0.2 * (first + second))
        self.assertTrue(all(torch.count_nonzero(g).item() == 0 for g in meta.grad_buffer))
        self.assert_state_equal(learner.low_rank_state, meta.low_rank_transport.dump_state())

    def saved_learner(self, directory):
        meta = self.make_meta()
        meta.should_train, meta.T = False, 2
        with torch.no_grad():
            for index, gate in enumerate(meta.gate_logits):
                gate.fill_(-2.0 + index)
            for u in meta.low_rank_transport.U.values():
                u.fill_(0.12)
            for v in meta.low_rank_transport.V.values():
                v.add_(0.2)
        source = meta.meta_fit(mock.Mock(), mock.Mock())
        source.save(directory)
        loaded = LRSGMAML.MyLearner()
        loaded.load(directory)
        return source, loaded

    def test_checkpoint_roundtrip_any_way_and_task_reset_preserve_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            source, loaded = self.saved_learner(directory)
            expected_files = {"model_args.pickle", "model_state.pickle", "weights.pickle",
                              "maml_params.pickle", "gate_logits.pickle",
                              "low_rank_transport.pickle", "low_rank_diagnostics.json"}
            self.assertEqual({p.name for p in Path(directory).iterdir()}, expected_files)
            self.assert_tensors_equal(loaded.weights, source.weights)
            self.assert_tensors_equal(loaded.gate_logits, source.gate_logits)
            self.assert_state_equal(loaded.low_rank_transport.dump_state(), source.low_rank_state)
            self.assertTrue(all(not p.requires_grad for p in loaded.gate_logits))
            self.assertTrue(all(not p.requires_grad for p in loaded.low_rank_transport.parameters()))
            weights_before = [p.detach().clone() for p in loaded.weights]
            gates_before = [p.detach().clone() for p in loaded.gate_logits]
            factors_before = loaded.low_rank_transport.dump_state()
            original_update = LRSGMAML.update_weights
            for ways, shots in ((2, 1), (5, 2), (2, 2)):
                with self.subTest(ways=ways, shots=shots):
                    starting_weights = []

                    def record_update(*args, **kwargs):
                        if not starting_weights:
                            starting_weights.extend(p.detach().clone() for p in args[0])
                        return original_update(*args, **kwargs)

                    episode = task(ways, shots)
                    with mock.patch.object(LRSGMAML, "update_weights", side_effect=record_update):
                        predictor = loaded.fit((*episode.support_set, ways, shots))
                    self.assert_tensors_equal(starting_weights[:-2], weights_before[:-2])
                    probabilities = predictor.predict(episode.query_set[0])
                    self.assertEqual(probabilities.shape, (ways * shots, ways))
                    self.assertTrue(np.isfinite(probabilities).all())
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
                    self.assertEqual(predictor.weights[-2].shape, (ways, 3))
                    self.assertEqual(predictor.weights[-1].shape, (ways,))
            self.assert_tensors_equal(loaded.weights, weights_before)
            self.assert_tensors_equal(loaded.gate_logits, gates_before)
            self.assert_state_equal(loaded.low_rank_transport.dump_state(), factors_before)

    def test_checkpoint_rejects_missing_incompatible_and_wrong_factor_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.saved_learner(directory)
            checkpoint = Path(directory) / "low_rank_transport.pickle"
            good = source.low_rank_state
            wrong_rank = copy.deepcopy(good)
            wrong_rank["rank"] = 0
            wrong_version = copy.deepcopy(good)
            wrong_version["format_version"] += 1
            wrong_type = copy.deepcopy(good)
            wrong_type["transport_type"] = "unrecognized_transform"
            wrong_names = copy.deepcopy(good)
            wrong_names["weight_names"] = tuple(reversed(good["weight_names"]))
            wrong_shape = copy.deepcopy(good)
            first_name = next(iter(wrong_shape["layers"]))
            wrong_shape["layers"][first_name]["U"] = torch.ones(1, 1)
            for bad in ({}, wrong_rank, wrong_version, wrong_type, wrong_names, wrong_shape):
                with self.subTest(keys=list(bad)):
                    with checkpoint.open("wb") as handle:
                        pickle.dump(bad, handle)
                    with self.assertRaises((ValueError, TypeError)):
                        LRSGMAML.MyLearner().load(directory)
            checkpoint.unlink()
            with self.assertRaisesRegex(Exception, "low_rank_transport"):
                LRSGMAML.MyLearner().load(directory)

    def test_factor_creation_and_loading_preserve_global_cpu_cuda_rng(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=str(device)):
                # Model initialization has its existing RNG cost; snapshot afterwards.
                network = TinyConvNetwork(3, device).to(device)
                cpu_before = torch.get_rng_state().clone()
                cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                python_before = random.getstate()
                numpy_before = np.random.get_state()
                first = LRSGMAML.LowRankTransport(network, rank=2, seed=98)
                checkpoint = first.dump_state()
                second = LRSGMAML.LowRankTransport(network, rank=2, seed=7)
                second.load_state(checkpoint, trainable=False)
                self.assertTrue(torch.equal(cpu_before, torch.get_rng_state()))
                cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                self.assert_tensors_equal(cuda_after, cuda_before)
                self.assertEqual(random.getstate(), python_before)
                numpy_after = np.random.get_state()
                self.assertEqual(numpy_after[0], numpy_before[0])
                np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
                self.assertEqual(numpy_after[2:], numpy_before[2:])
                self.assert_state_equal(second.dump_state(), checkpoint)

    def test_diagnostics_are_detached_rng_safe_and_explicit_about_zero_gradients(self):
        network = TinyConvNetwork(3, torch.device("cpu"))
        transport = LRSGMAML.LowRankTransport(network, rank=2)
        weights = list(network.parameters())
        gates = [nn.Parameter(torch.tensor(4.0)) for _ in weights]
        gradients = [torch.linspace(-30, 20, p.numel()).reshape_as(p) for p in weights]
        reference = LRSGMAML.update_weights(
            weights, gradients, 1.5, 0.13, gates, transport.weight_names, transport)
        rng_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        observer = LRSGMAML.TransportDiagnostics(transport, gates)
        with mock.patch.object(observer, "record", wraps=observer.record) as record:
            actual = LRSGMAML.update_weights(
                weights, gradients, 1.5, 0.13, gates, transport.weight_names, transport, observer)
            self.assert_tensors_equal(actual, reference)
            self.assertEqual(record.call_count, len(transport.conv_names))
            for call in record.call_args_list:
                self.assertTrue(all(not value.requires_grad and value.grad_fn is None
                                    for value in call.args[1:]))
                self.assertLessEqual(call.args[1].abs().max().item(), 1.5)
        LRSGMAML.update_weights(
            weights, [torch.zeros_like(g) for g in gradients], 1.5, 0.13,
            gates, transport.weight_names, transport, observer)
        summary = observer.summary()
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        self.assert_tensors_equal(cuda_after, cuda_before)
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        # Persistent measurements contain JSON scalars, not tensors or graphs.
        json.dumps(observer.layers, allow_nan=False)
        json.dumps(summary, allow_nan=False)
        self.assertEqual(summary["gate_statistics"]["count"], len(gates))
        for layer in summary["layers"].values():
            self.assertEqual(layer["observations"], 2)
            self.assertEqual(layer["zero_input_norm"], 1)
            self.assertEqual(layer["zero_scalar_norm"], 1)
            self.assertEqual(layer["zero_transformed_norm"], 1)
            self.assertEqual(layer["cosine_count"], 1)
            self.assertEqual(layer["correction_ratio_mean"], 0.0)
            self.assertAlmostEqual(layer["gradient_cosine_mean"], 1.0, places=6)

    def test_validation_best_snapshot_pairs_weights_gates_and_factors(self):
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
                for u in meta.low_rank_transport.U.values():
                    u.fill_(marker / 10)
                for v in meta.low_rank_transport.V.values():
                    v.fill_(-marker / 10)
            predictions = labels.clone()
            predictions[num_correct:] = 1 - predictions[num_correct:]
            logits = F.one_hot(predictions, num_classes=2).float()
            with mock.patch.object(meta, "compute_out_and_loss", return_value=(logits, None)):
                meta.meta_valid(lambda _: [episode])
            if marker in (1.0, 4.0):
                previous_best = ([p.detach().clone() for p in meta.weights],
                                 [p.detach().clone() for p in meta.gate_logits],
                                 meta.low_rank_transport.dump_state())
            self.assert_tensors_equal(meta.best_state, previous_best[0])
            self.assert_tensors_equal(meta.best_gate_logits, previous_best[1])
            self.assert_state_equal(meta.best_low_rank_state, previous_best[2])
        with torch.no_grad():
            meta.weights[0].add_(10)
            meta.gate_logits[0].add_(10)
            next(iter(meta.low_rank_transport.U.values())).add_(10)
        meta.should_train = False
        learner = meta.meta_fit(mock.Mock(), mock.Mock())
        self.assert_tensors_equal(learner.weights, previous_best[0])
        self.assert_tensors_equal(learner.gate_logits, previous_best[1])
        self.assert_state_equal(learner.low_rank_state, previous_best[2])


if __name__ == "__main__":
    unittest.main()
