"""Short CPU checks: python -m unittest discover -s baselines/tcsgmaml/tests -v."""

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

ALIASES = {
    "sgmaml": ("weight_names", "network", "api", "helpers_sgmaml"),
    "tcsgmaml": ("weight_names", "network", "api", "task_conditioning",
                 "helpers_tcsgmaml"),
}


def load_baseline(name):
    """Resolve submission-local imports without polluting other baselines."""
    root = BASELINES / name
    aliases = ALIASES[name]
    previous = {alias: sys.modules.get(alias) for alias in aliases}
    loaded = {}
    try:
        for alias in (*aliases, "model"):
            spec = importlib.util.spec_from_file_location(
                "_test_" + name + "_" + alias, root / (alias + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[alias] = module
            if alias != "model":
                sys.modules[alias] = module
        return loaded
    finally:
        for alias, module in previous.items():
            if module is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = module


_CONDITIONED = load_baseline("tcsgmaml")
TCSGMAML = _CONDITIONED["model"]
CONDITIONING = _CONDITIONED["task_conditioning"]
SGMAML = load_baseline("sgmaml")["model"]


class TinyNetwork(nn.Module):
    """Nonlinear fast-weight model with the same final weight/bias contract."""

    def __init__(self, num_classes, dev, num_blocks=18, pretrained=False,
                 img_size=128, **kwargs):
        super().__init__()
        self.dev = dev
        self.num_blocks = num_blocks
        self.in_features = 2
        self.encoder = nn.Linear(3, 2)
        self.bn = nn.BatchNorm1d(2, momentum=1)
        # Mirror the real network, whose classifier lives at "model.out".
        self.model = nn.ModuleDict({"out": nn.Linear(2, num_classes)})
        self.criterion = nn.CrossEntropyLoss()

    def forward_weights(self, x, weights, embedding=False):
        z = F.linear(x, weights[0], weights[1])
        z = torch.tanh(F.batch_norm(
            z, torch.zeros(2, device=z.device), torch.ones(2, device=z.device),
            weights[2], weights[3], momentum=1, training=True))
        if embedding:
            return z
        return F.linear(z, weights[-2], weights[-1])

    def compute_in_features(self, x):
        return torch.tanh(self.bn(self.encoder(x)))

    def modify_out_layer(self, num_classes):
        self.model.out = nn.Linear(2, num_classes).to(self.dev)
        nn.init.zeros_(self.model.out.bias)

    def load_params(self, state):
        self.load_state_dict({k: v for k, v in state.items()
                              if not k.startswith("model.out.")}, strict=False)


def task(ways=3, shots=2, offset=0.0):
    labels = torch.arange(ways).repeat_interleave(shots)
    x = torch.linspace(-1.1, 1.3, labels.numel() * 3).reshape(-1, 3)
    query_labels = labels.roll(1)
    return SimpleNamespace(
        num_ways=ways, support_set=(x + offset, labels, labels),
        query_set=(x.flip(0) - offset, query_labels, query_labels))


class TaskConditionedGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)
        for module, name in ((TCSGMAML, "ResNet"), (CONDITIONING, "ResNet")):
            patch = mock.patch.object(module, name, TinyNetwork)
            patch.start()
            self.addCleanup(patch.stop)
        self.device_patch = mock.patch.object(
            TCSGMAML.MyMetaLearner, "get_device",
            return_value=torch.device("cpu"))
        self.device_patch.start()
        self.addCleanup(self.device_patch.stop)
        TCSGMAML._ENCODER_CACHE.clear()
        TCSGMAML._ENCODER_PACKAGE_CACHE.clear()
        self.addCleanup(TCSGMAML._ENCODER_CACHE.clear)
        self.addCleanup(TCSGMAML._ENCODER_PACKAGE_CACHE.clear)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.logger = SimpleNamespace(log=mock.Mock())
        self.checkpoint = self.write_encoder_checkpoint()

    def write_encoder_checkpoint(self, name="encoder_weights.pickle",
                                 num_classes=3):
        """Store a weight list exactly as Learner.save writes weights.pickle."""
        torch.manual_seed(5)
        source = TinyNetwork(num_classes, torch.device("cpu"))
        with torch.no_grad():
            source.encoder.weight.normal_()
            source.bn.weight.normal_(1.0, 0.2)
        path = self.root / name
        with path.open("wb") as handle:
            pickle.dump([p.detach().clone() for p in source.parameters()],
                        handle)
        return path

    def config(self, **encoder_overrides):
        config = json.loads((BASELINES / "tcsgmaml/config.json").read_text())
        config["task_encoder"]["checkpoint"] = str(self.checkpoint)
        config["task_encoder"].update(encoder_overrides)
        return config

    def make_meta(self, **encoder_overrides):
        with mock.patch.object(TCSGMAML, "load_config",
                               return_value=self.config(**encoder_overrides)):
            return TCSGMAML.MyMetaLearner(3, 9, self.logger)

    def support(self, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(6, 3, generator=generator)

    def assert_tensors_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_config_matches_sgmaml_apart_from_task_conditioning(self):
        original = json.loads((BASELINES / "sgmaml/config.json").read_text())
        conditioned = json.loads(
            (BASELINES / "tcsgmaml/config.json").read_text())
        encoder = conditioned.pop("task_encoder")
        gate_net = conditioned.pop("gate_net")
        self.assertEqual(conditioned, original)
        # The checkpoint path is specific to whoever runs this, so only its
        # presence is checked here. Refusing an absent one is covered by
        # test_encoder_refuses_to_guess_a_checkpoint.
        self.assertIn("checkpoint", encoder)
        self.assertEqual({key: value for key, value in encoder.items()
                          if key != "checkpoint"},
                         {"forward": "fast_weights", "num_blocks": 18,
                          "img_size": 128})
        self.assertEqual(gate_net, {"hidden_size": 128, "input_norm": "none",
                                    "delta_scale": 1.0})

    def test_initialization_reproduces_sgmaml_gates_for_every_task(self):
        meta = self.make_meta()
        self.assertEqual(meta.gate_init_logit, 4.0)
        for seed in (1, 2):
            gates = torch.stack(
                meta.task_gate_logits(self.support(seed), detach=True))
            torch.testing.assert_close(
                gates, torch.full_like(gates, 4.0), rtol=0, atol=0)
        self.assertEqual(meta.gate_net.out.weight.abs().sum().item(), 0.0)
        self.assertEqual(meta.gate_net.out.bias.abs().sum().item(), 0.0)

    def test_outer_groups_hold_three_learnables_and_exclude_the_encoder(self):
        meta = self.make_meta()
        expected = (meta.weights + list(meta.gate_logits)
                    + list(meta.gate_net.parameters()))
        self.assertEqual([id(p) for p in meta.meta_parameters],
                         [id(p) for p in expected])
        optimizer_params = [p for group in meta.optimizer.param_groups
                            for p in group["params"]]
        self.assertEqual([id(p) for p in optimizer_params],
                         [id(p) for p in expected])
        self.assertEqual([p.shape for p in meta.grad_buffer],
                         [p.shape for p in expected])
        self.assertEqual(len(meta.gate_logits), len(meta.weights))
        self.assertEqual(meta.gate_net.num_gates, len(meta.weights))
        self.assertEqual(meta.gate_net.in_features,
                         meta.task_encoder.out_features)
        encoder_ids = {id(p) for p in meta.task_encoder.network.parameters()}
        encoder_ids |= {id(w) for w in meta.task_encoder.weights}
        self.assertFalse(encoder_ids & {id(p) for p in meta.meta_parameters})
        self.assertFalse(any(p.requires_grad for p in
                             meta.task_encoder.network.parameters()))
        self.assertFalse(any(w.requires_grad
                             for w in meta.task_encoder.weights))
        self.assertFalse(meta.task_encoder.network.training)

    def test_construction_leaves_the_global_rng_stream_untouched(self):
        torch.manual_seed(31)
        self.make_meta()
        conditioned = torch.randn(5)
        torch.manual_seed(31)
        with mock.patch.object(SGMAML, "ResNet", TinyNetwork), \
                mock.patch.object(SGMAML.MyMetaLearner, "get_device",
                                  return_value=torch.device("cpu")):
            SGMAML.MyMetaLearner(3, 9, self.logger)
        torch.testing.assert_close(conditioned, torch.randn(5),
                                   rtol=0, atol=0)

    def test_query_loss_trains_shared_logits_and_gatenet_first_order(self):
        meta = self.make_meta()
        meta.T = 2
        with torch.no_grad():
            meta.gate_net.out.weight.normal_(0.0, 0.5)
            meta.gate_net.out.bias.normal_(0.0, 0.5)
        episode = task()
        xs, ys, _ = episode.support_set
        xq, yq, _ = episode.query_set

        embedding = meta.task_encoder.embed(xs)
        self.assertFalse(embedding.requires_grad)
        self.assertEqual(embedding.shape, (meta.task_encoder.out_features,))

        gates = meta.task_gate_logits(xs)
        self.assertTrue(all(gate.requires_grad for gate in gates))
        _, loss = meta.compute_out_and_loss(
            meta.meta_learner, [w.clone() for w in meta.weights], gates,
            xs, ys, xq, yq, 3, True)
        gradients = torch.autograd.grad(loss, meta.meta_parameters)
        shared = gradients[len(meta.weights):
                           len(meta.weights) + meta.num_gates]
        gate_net = gradients[len(meta.weights) + meta.num_gates:]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().item() > 0 for g in shared))
        self.assertTrue(any(g.abs().sum().item() > 0 for g in gate_net))
        self.assertFalse(meta.second_order)
        self.assertTrue(all(w.grad is None
                            for w in meta.task_encoder.weights))

    def test_one_gate_set_per_episode_reused_by_every_inner_step(self):
        meta = self.make_meta()
        meta.T = 3
        with torch.no_grad():
            meta.gate_net.out.weight.normal_(0.0, 0.5)
        episode = task()
        xs, ys, _ = episode.support_set
        xq, yq, _ = episode.query_set
        gates = meta.task_gate_logits(xs, detach=True)
        with mock.patch.object(TCSGMAML, "update_weights",
                               wraps=TCSGMAML.update_weights) as update:
            meta.compute_out_and_loss(
                meta.meta_learner, [w.clone() for w in meta.weights], gates,
                xs, ys, xq, yq, 3, False, True)
        self.assertEqual(update.call_count, meta.T)
        for call in update.call_args_list:
            self.assertIs(call.args[-1], gates)
        first = torch.stack(meta.task_gate_logits(self.support(11),
                                                  detach=True))
        second = torch.stack(meta.task_gate_logits(self.support(12),
                                                   detach=True))
        self.assertFalse(torch.allclose(first, second))

    def test_validation_pairs_the_best_state_and_records_gate_spread(self):
        meta = self.make_meta()
        with torch.no_grad():
            meta.gate_net.out.weight.normal_(0.0, 0.5)
        episode = task(ways=2, shots=1)
        labels = episode.query_set[1]
        previous_best = None
        for marker, num_correct in ((1.0, 1), (2.0, 0), (3.0, 2)):
            with torch.no_grad():
                for p in meta.weights:
                    p.fill_(marker)
                for gate in meta.gate_logits:
                    gate.fill_(-marker)
                meta.gate_net.out.bias.fill_(marker)
            predictions = labels.clone()
            predictions[num_correct:] = 1 - predictions[num_correct:]
            logits = F.one_hot(predictions, num_classes=2).float()
            with mock.patch.object(meta, "compute_out_and_loss",
                                   return_value=(logits, None)):
                meta.meta_valid(lambda _: [episode, episode])
            if marker in (1.0, 3.0):
                previous_best = (
                    [p.detach().clone() for p in meta.weights],
                    [p.detach().clone() for p in meta.gate_logits],
                    meta.gate_net_state())
            self.assert_tensors_equal(meta.best_state, previous_best[0])
            self.assert_tensors_equal(meta.best_gate_logits, previous_best[1])
            self.assertEqual(
                {k: v.tolist() for k, v in meta.best_gate_net_state.items()},
                {k: v.tolist() for k, v in previous_best[2].items()})
        self.assertEqual(len(meta.gate_summaries), 3)
        summary = meta.gate_summaries[-1]
        self.assertEqual(summary["tasks"], 2)
        self.assertEqual(summary["tensors"], len(meta.weights))
        for key in ("gate_mean", "gate_min", "gate_max",
                    "across_task_std_mean", "across_tensor_std"):
            self.assertTrue(np.isfinite(summary[key]))
        self.assertEqual(summary["across_task_std_mean"], 0.0)

    def saved_learner(self):
        meta = self.make_meta()
        meta.should_train = True
        meta.T, meta.train_tasks, meta.val_after = 2, 2, 99
        meta.optimizer = torch.optim.SGD(meta.meta_parameters, lr=0.2)
        with torch.no_grad():
            meta.gate_net.out.weight.normal_(0.0, 0.5)
            meta.gate_net.out.bias.normal_(0.0, 0.5)
        # Both episodes keep the training number of ways, as the generator
        # does, but differ in support size so that their gates differ.
        episodes = [task(ways=3, shots=2), task(ways=3, shots=3)]
        source = meta.meta_fit(lambda count: iter(episodes), mock.Mock())
        directory = self.root / "model"
        directory.mkdir(exist_ok=True)
        source.save(str(directory))
        loaded = TCSGMAML.MyLearner()
        loaded.load(str(directory))
        return meta, source, loaded, directory

    def test_checkpoint_roundtrip_keeps_all_three_components_fixed(self):
        meta, source, loaded, directory = self.saved_learner()
        self.assertEqual({p.name for p in directory.iterdir()}, {
            "model_args.pickle", "model_state.pickle", "weights.pickle",
            "maml_params.pickle", "gate_logits.pickle", "gate_net.pickle",
            "task_encoder.pickle", "gate_stats.pickle"})
        self.assert_tensors_equal(loaded.weights, source.weights)
        self.assert_tensors_equal(loaded.gate_logits, source.gate_logits)
        self.assertTrue(all(not p.requires_grad for p in loaded.gate_logits))
        self.assertTrue(all(not p.requires_grad
                            for p in loaded.gate_net.parameters()))
        self.assertFalse(loaded.gate_net.training)
        for name, value in meta.gate_net_state().items():
            torch.testing.assert_close(
                loaded.gate_net.state_dict()[name], value, rtol=0, atol=0)
        self.assert_tensors_equal(loaded.task_encoder.weights,
                                  meta.task_encoder.weights)
        self.assertEqual(loaded.task_encoder.signature["fingerprint"],
                         meta.task_encoder.signature["fingerprint"])

        gates_before = [p.clone() for p in loaded.gate_logits]
        weights_before = [p.detach().clone() for p in loaded.weights]
        seen = []
        with mock.patch.object(TCSGMAML, "update_weights",
                               wraps=TCSGMAML.update_weights) as update:
            for ways, shots in ((2, 1), (5, 2)):
                with self.subTest(ways=ways, shots=shots):
                    episode = task(ways, shots)
                    predictor = loaded.fit((*episode.support_set, ways, shots))
                    probabilities = predictor.predict(episode.query_set[0])
                    self.assertEqual(probabilities.shape, (ways * shots, ways))
                    self.assertTrue(np.isfinite(probabilities).all())
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0,
                                               atol=1e-6)
                    self.assertEqual(predictor.weights[-2].shape, (ways, 2))
                    self.assertEqual(predictor.weights[-1].shape, (ways,))
        self.assertEqual(update.call_count, 2 * loaded.T)
        for call in update.call_args_list:
            seen.append(torch.stack(call.args[-1]).detach())
            self.assertEqual(len(call.args[-1]), len(loaded.weights))
        # Every inner step of one task shares its gates, and a new support set
        # produces a new set of them.
        torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
        self.assertFalse(torch.allclose(seen[0], seen[-1]))
        self.assert_tensors_equal(loaded.gate_logits, gates_before)
        self.assert_tensors_equal(loaded.weights, weights_before)

    def test_checkpoint_rejects_missing_or_mismatched_components(self):
        _, source, _, directory = self.saved_learner()
        for name in ("gate_net.pickle", "task_encoder.pickle"):
            with self.subTest(name=name):
                path = directory / name
                kept = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(Exception, name.split(".")[0]):
                    TCSGMAML.MyLearner().load(str(directory))
                path.write_bytes(kept)
        package = pickle.loads((directory / "gate_net.pickle").read_bytes())
        for broken in ({"config": package["config"], "state_dict": None},
                       {"config": dict(package["config"], num_gates=3),
                        "state_dict": package["state_dict"]}):
            with (directory / "gate_net.pickle").open("wb") as handle:
                pickle.dump(broken, handle)
            TCSGMAML._ENCODER_CACHE.clear()
            with self.assertRaises(ValueError):
                TCSGMAML.MyLearner().load(str(directory))

    def test_encoder_refuses_to_guess_a_checkpoint(self):
        for checkpoint in (None, "", "   "):
            with self.subTest(checkpoint=checkpoint):
                with self.assertRaisesRegex(ValueError, "task_encoder"):
                    CONDITIONING.FrozenTaskEncoder.from_checkpoint(
                        checkpoint, torch.device("cpu"))
        with self.assertRaises(FileNotFoundError):
            CONDITIONING.FrozenTaskEncoder.from_checkpoint(
                str(self.root / "absent.pickle"), torch.device("cpu"))
        wrong = self.root / "wrong_shape.pickle"
        with wrong.open("wb") as handle:
            pickle.dump([torch.zeros(4, 4)] * 6, handle)
        with self.assertRaises(ValueError):
            CONDITIONING.FrozenTaskEncoder.from_checkpoint(
                str(wrong), torch.device("cpu"))
        short = self.root / "wrong_count.pickle"
        with short.open("wb") as handle:
            pickle.dump([torch.zeros(2, 3)], handle)
        with self.assertRaises(ValueError):
            CONDITIONING.FrozenTaskEncoder.from_checkpoint(
                str(short), torch.device("cpu"))

    def write_state_dict(self, name, drop=()):
        """Store a state dict checkpoint, optionally with keys removed."""
        torch.manual_seed(9)
        source = TinyNetwork(2, torch.device("cpu"))
        path = self.root / name
        with path.open("wb") as handle:
            pickle.dump({key: value.clone() for key, value
                         in source.state_dict().items()
                         if key not in drop}, handle)
        return path

    def test_state_dict_must_define_every_encoder_parameter(self):
        incomplete = self.write_state_dict("no_bn_weight.pickle",
                                           drop=("bn.weight",))
        with self.assertRaisesRegex(ValueError, "bn.weight"):
            CONDITIONING.FrozenTaskEncoder.from_checkpoint(
                str(incomplete), torch.device("cpu"))
        # The classifier is the one part the task embedding never uses, so a
        # checkpoint is allowed to leave it out.
        headless = self.write_state_dict(
            "no_classifier.pickle",
            drop=("model.out.weight", "model.out.bias"))
        encoder = CONDITIONING.FrozenTaskEncoder.from_checkpoint(
            str(headless), torch.device("cpu"))
        self.assertEqual(encoder.embed(self.support(4)).shape,
                         (encoder.out_features,))

    def test_module_eval_refuses_incomplete_or_untouched_statistics(self):
        for name, drop in (("untouched_bn.pickle", ()),
                           ("missing_bn.pickle",
                            ("bn.running_mean", "bn.running_var"))):
            with self.subTest(name=name):
                path = self.write_state_dict(name, drop=drop)
                with self.assertRaisesRegex(ValueError, "BatchNorm"):
                    CONDITIONING.FrozenTaskEncoder.from_checkpoint(
                        str(path), torch.device("cpu"),
                        forward_mode="module_eval")
                encoder = CONDITIONING.FrozenTaskEncoder.from_checkpoint(
                    str(path), torch.device("cpu"),
                    forward_mode="fast_weights")
                self.assertFalse(encoder.has_bn_statistics)
                self.assertEqual(encoder.embed(self.support(3)).shape,
                                 (encoder.out_features,))

        trained = self.write_state_dict("trained_bn.pickle")
        state = pickle.loads(trained.read_bytes())
        state["bn.running_var"] = torch.full_like(state["bn.running_var"], 2.0)
        with trained.open("wb") as handle:
            pickle.dump(state, handle)
        encoder = CONDITIONING.FrozenTaskEncoder.from_checkpoint(
            str(trained), torch.device("cpu"), forward_mode="module_eval")
        self.assertTrue(encoder.has_bn_statistics)
        self.assertEqual(encoder.embed(self.support(3)).shape,
                         (encoder.out_features,))

    def test_gate_summary_separates_task_spread_from_tensor_spread(self):
        logits = np.array([[1.0, -1.0], [1.0, -1.0]])
        summary = CONDITIONING.summarize_gates(logits)
        self.assertEqual(summary["across_task_std_mean"], 0.0)
        self.assertEqual(summary["across_task_range_max"], 0.0)
        self.assertGreater(summary["across_tensor_std"], 0.0)
        moving = CONDITIONING.summarize_gates(np.array([[1.0], [-1.0]]))
        self.assertGreater(moving["across_task_std_mean"], 0.0)
        self.assertIn("gates", CONDITIONING.format_gate_summary(summary, 10))


if __name__ == "__main__":
    unittest.main()
