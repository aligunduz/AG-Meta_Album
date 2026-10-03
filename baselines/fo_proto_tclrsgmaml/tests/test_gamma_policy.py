"""Small CPU gamma-policy checks: no optimizer, datasets, or W&B run."""
import copy
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
from helpers_fo_proto_tclrsgmaml import adapt, adaptation_gamma, prototype_head
from task_transport import TaskConditionedTransport
from eval_beta_sweep import evaluate_task, scaled_low_rank


class Tiny(nn.Module):
    in_features = 4

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)

    def forward_weights(self, x, weights, embedding=False):
        features = F.linear(x, weights[0], weights[1]).tanh()
        return features if embedding else F.linear(features, weights[-2], weights[-1])


class GammaPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(17)
        self.config = baseline.read_config()
        self.config["lrsg"]["beta"] = .75
        self.config["method_config"].update(inner_steps=2, encoder_lr=.07,
                                             classifier_lr=.13, grad_clip=.005)
        self.encoder = Tiny().double().eval()
        self.transport = TaskConditionedTransport(self.encoder, self.config).eval()
        with torch.no_grad():
            self.transport.u["0"].normal_(0, .7)
            self.transport.gate_net.out.weight.normal_(0, .5)
            self.transport.gate_net.out.bias.normal_(0, .3)
        self.x = torch.randn(6, 3, dtype=torch.float64)
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1])
        self.learner = baseline.MyLearner()
        self.learner.learner = self.encoder
        self.learner.transport = self.transport
        self.learner.config = self.config
        self.learner.dev = torch.device("cpu")
        self.learner.model_args = dict(num_classes=2, dev="cpu")
        self.learner.state = baseline.snapshot(self.encoder, self.transport)
        self.learner.best_score = .5

    def weights(self, phase, config=None, x=None, labels=None):
        return adapt(self.encoder, list(self.encoder.parameters()),
                     self.x if x is None else x,
                     self.labels if labels is None else labels,
                     self.config["method_config"] if config is None else config,
                     transport=self.transport, phase=phase)

    def assert_weights_equal(self, a, b):
        for actual, expected in zip(a, b, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_phase_defaults_and_override_match_original_beta_sweep_all_conditions(self):
        for ways in (2, 7, 20):
            for shots in (1, 3, 20):
                with self.subTest(ways=ways, shots=shots):
                    labels = torch.arange(ways).repeat_interleave(shots)
                    x = torch.randn(ways * shots, 3, dtype=torch.float64)
                    train = self.weights("train", x=x, labels=labels)
                    # Independent old experiment: scale beta and use gamma=1.
                    with scaled_low_rank(self.transport, 2):
                        oracle = self.weights("train", x=x, labels=labels)
                    for phase in ("validation", "test"):
                        with torch.no_grad():
                            evaluated = self.weights(phase, x=x, labels=labels)
                        self.assert_weights_equal(evaluated, oracle)
                        legacy = self.weights(phase, dict(self.config["method_config"],
                                                         eval_gamma=1), x, labels)
                        self.assert_weights_equal(legacy, train)
        # A nonzero operator makes accidental gamma=1/4 distinguishable.
        self.assertFalse(torch.equal(self.weights("train")[0], self.weights("test")[0]))

    def test_only_clipped_low_rank_correction_is_scaled_in_one_step(self):
        config = dict(self.config["method_config"], inner_steps=1)
        weights = list(self.encoder.parameters())
        features = self.encoder.forward_weights(self.x, weights, embedding=True)
        fast = [w.clone() for w in weights] + list(prototype_head(features, self.labels))
        condition = self.transport.condition(features.mean(0).detach())
        loss = F.cross_entropy(self.encoder.forward_weights(self.x, fast), self.labels)
        grads = [g.clamp(-config["grad_clip"], config["grad_clip"])
                 for g in torch.autograd.grad(loss, fast, retain_graph=True)]
        actual = self.weights("test", config)
        for i, (name, _) in enumerate(self.encoder.named_parameters()):
            key = self.transport.indices[name]
            transformed = self.transport.logits[key].sigmoid() * grads[i]
            if key in self.transport.u:
                projection = self.transport.v[key].T @ grads[i]
                projection = (1 + condition[1][key]).unsqueeze(1) * projection
                transformed = transformed + (self.transport.beta * 2) * (
                    self.transport.u[key] @ projection)
            torch.testing.assert_close(actual[i], fast[i] - config["encoder_lr"] * transformed,
                                       rtol=0, atol=0)
        for i in (-2, -1):
            torch.testing.assert_close(actual[i], fast[i] - config["classifier_lr"] * grads[i],
                                       rtol=0, atol=0)
        self.assert_weights_equal(actual[-2:], self.weights("train", config)[-2:])

    def test_public_callers_select_phase_even_when_support_gradients_are_enabled(self):
        task = SimpleNamespace(support_set=(self.x, self.labels, None),
                               query_set=(self.x + .1, self.labels, None), num_ways=2)
        meta = baseline.MyMetaLearner.__new__(baseline.MyMetaLearner)
        meta.config, meta.params = self.config, self.config["method_config"]
        meta.meta_learner, meta.transport = self.encoder, self.transport
        meta.weights = list(self.encoder.parameters())
        meta.meta_parameters = meta.weights + list(self.transport.parameters())
        meta.dev, meta.best_score, meta.best_state = torch.device("cpu"), -float("inf"), None
        meta.log = lambda *a, **kw: None

        def stop_before_training(*args, **kwargs):
            self.assertEqual(kwargs["phase"], "train")
            self.assertEqual(adaptation_gamma(args[4], kwargs["phase"]), 1)
            raise RuntimeError("checked train dispatch before training")

        with patch.object(baseline, "adapt", side_effect=stop_before_training):
            with self.assertRaisesRegex(RuntimeError, "checked train dispatch"):
                meta.meta_fit(lambda count: iter([task]), None)

        seen = []
        original = self.transport.transport_gradient

        def observe(name, gradient, conditioning, *, gamma):
            seen.append((gamma, torch.is_grad_enabled()))
            return original(name, gradient, conditioning, gamma=gamma)

        with patch.object(self.transport, "transport_gradient", side_effect=observe):
            meta.meta_valid(lambda count: iter([task]))
            self.assertIsNotNone(meta.best_state)
            # Deliberately wrong mode: fit's explicit test phase still wins.
            self.encoder.train()
            self.transport.train()
            for context in (torch.no_grad(), torch.enable_grad()):
                with context:
                    self.learner.fit((self.x, self.labels, None, 2, 3))
        self.assertTrue(seen)
        self.assertTrue(all(gamma == 2 and enabled for gamma, enabled in seen))

    def test_old_checkpoint_and_runtime_override_without_retraining(self):
        old = copy.deepcopy(self.config)
        old["method_config"].pop("train_gamma")
        old["method_config"].pop("eval_gamma")
        self.learner.config = old
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            self.learner.save(directory)
            before = (Path(directory) / "max-va.pth").read_bytes()
            with patch.object(baseline, "make_encoder", side_effect=lambda args: Tiny().double()), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                for runtime, override, expected in ((old, None, 2), (self.config, 1, 1),
                        (dict(self.config, method_config=dict(self.config["method_config"],
                                                            eval_gamma=1)), None, 1)):
                    with patch.object(baseline, "read_config", return_value=runtime):
                        loaded = baseline.MyLearner()
                        loaded.load(directory, eval_gamma=override)
                    self.assertEqual(loaded.config["method_config"]["train_gamma"], 1)
                    self.assertEqual(loaded.config["method_config"]["eval_gamma"], expected)
                    self.assertEqual(loaded.transport.architecture(), self.learner.state["architecture"])
                    self.assertEqual(loaded.transport.beta, .75)
                    actual = loaded.fit((self.x, self.labels, None, 2, 3)).weights
                    self.assert_weights_equal(actual, self.weights("test", dict(
                        old["method_config"], eval_gamma=expected)))
                self.assertEqual((Path(directory) / "max-va.pth").read_bytes(), before)
                # Even a saved gamma=1 must not defeat the current default=2.
                self.learner.config = baseline.gamma_defaults(old)
                self.learner.config["method_config"]["eval_gamma"] = 1
                self.learner.save(directory)
                loaded = baseline.MyLearner()
                loaded.load(directory)
                self.assertEqual(loaded.config["method_config"]["eval_gamma"], 2)
            self.assertNotIn("eval_gamma", old["method_config"])

    def test_sweep_gamma_two_matches_default_and_restores_policy_without_double_scaling(self):
        task = SimpleNamespace(support_set=(self.x, self.labels, None),
                               query_set=(self.x + .1, self.labels, None),
                               num_ways=2, num_shots=3, dataset="synthetic")
        original_config = self.config["method_config"]
        original_fit = self.learner.fit
        expected = original_fit((*task.support_set, 2, 3)).weights
        seen = []

        def observe(support):
            predictor = original_fit(support)
            seen.append(predictor.weights)
            return predictor

        with patch.object(self.learner, "fit", side_effect=observe):
            evaluate_task(self.learner, task, 1, (1., 2.))
        self.assert_weights_equal(seen[2], expected)
        self.assertIs(self.config["method_config"], original_config)
        self.assertEqual(original_config["eval_gamma"], 2)
        self.assertEqual(self.transport.beta, .75)
        calls = []

        def fail_in_gamma(support):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("stop")
            return original_fit(support)

        with patch.object(self.learner, "fit", side_effect=fail_in_gamma):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                evaluate_task(self.learner, task, 1, (1., 2.))
        self.assertIs(self.config["method_config"], original_config)
        self.assertEqual(self.transport.beta, .75)

    def test_missing_fields_and_invalid_policy(self):
        self.assertEqual(adaptation_gamma({}, "train"), 1)
        for phase in ("validation", "test"):
            self.assertEqual(adaptation_gamma({}, phase), 2)
        for value in (-1, float("nan"), float("inf"), True, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                baseline.validate_config(dict(self.config, method_config=dict(
                    self.config["method_config"], eval_gamma=value)))
        with self.assertRaises(ValueError):
            adaptation_gamma({}, "unknown")


if __name__ == "__main__":
    unittest.main()
