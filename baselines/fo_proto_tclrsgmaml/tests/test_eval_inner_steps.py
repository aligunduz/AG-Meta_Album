"""CPU synthetic evidence for diagnostics; not a trained Meta-Album experiment."""
import copy
import csv
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_eval_condition_controls import Tiny  # noqa: E402
import model as baseline  # noqa: E402
import eval_inner_steps as diagnostic  # noqa: E402
from task_transport import TaskConditionedTransport  # noqa: E402
from cdmetadl.ingestion.image_dataset import ImageDataset  # noqa: E402


class SyntheticDataset(ImageDataset):
    def __init__(self, name):
        self.name = name
        self.min_examples_per_class = 40
        self.idx_per_label = [np.arange(i * 40, (i + 1) * 40) for i in range(3)]

    def __getitem__(self, index):
        x = torch.tensor([np.sin(index), np.cos(index), index / 120], dtype=torch.float64)
        return x, torch.tensor(index // 40)


class InnerStepsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(17)
        config = baseline.read_config()
        config["task_conditioning"]["scalar_delta_scale"] = 0.0
        config["method_config"].update(inner_steps=5, encoder_lr=.07,
                                       classifier_lr=.13, grad_clip=.005)
        model = Tiny().double().eval()
        model.register_buffer("diagnostic_test_buffer", torch.tensor(2.))
        transport = TaskConditionedTransport(model, config).eval()
        with torch.no_grad():
            transport.u["0"].normal_(0, .7)
            transport.gate_net.out.weight.normal_(0, .5)
            transport.gate_net.out.bias.normal_(0, .3)
        learner = baseline.MyLearner()
        learner.learner, learner.transport, learner.config = model, transport, config
        learner.dev = torch.device("cpu")
        learner.best_score = .5
        learner.model_args = dict(num_classes=2, dev="cpu", num_blocks=18, pretrained=False)
        learner.state = baseline.snapshot(model, transport)
        self.learner = learner
        self.datasets = [SyntheticDataset("DOG"), SyntheticDataset("TEX")]
        self.task = next(self.tasks())

    def tasks(self, seed=93, count=2):
        return diagnostic.CompetitionDataLoader(self.datasets, diagnostic.TEST_EPISODES,
                                                seed, test_generator=True).generator(count)

    def fit(self, task=None):
        task = self.task if task is None else task
        return self.learner.fit((*task.support_set, task.num_ways, task.num_shots))

    def diagnostics(self, adapted):
        return diagnostic.support_diagnostics(self.learner, *self.task.support_set[:2],
                                               self.task.num_ways, adapted)

    def test_raw_clipped_norms_and_actual_first_step_residual(self):
        original_grad = torch.autograd.grad
        original_transport = self.learner.transport.transport_gradient
        raw, clipped, residuals = [], [], []

        def observe_grad(*args, **kwargs):
            result = original_grad(*args, **kwargs)
            if not raw:
                raw.extend(g.detach().clone() for g in result[:-2])
            return result

        def observe_transport(name, g, conditioning):
            result = original_transport(name, g, conditioning)
            if len(clipped) < len(self.learner.transport.names):
                clipped.append(g.detach().clone())
                key = self.learner.transport.indices[name]
                residuals.append((result - self.learner.transport.logits[key].sigmoid() * g).detach())
            return result

        with patch.object(torch.autograd, "grad", side_effect=observe_grad), \
                patch.object(self.learner.transport, "transport_gradient", side_effect=observe_transport):
            predictor = self.fit()
        values = self.diagnostics(predictor.weights)
        norm = lambda values: sum(v.double().square().sum() for v in values).sqrt().item()
        self.assertGreater(norm(raw), norm(clipped))
        self.assertAlmostEqual(values["encoder_grad_norm_raw_start"], norm(raw), places=14)
        self.assertAlmostEqual(values["encoder_grad_norm_clipped_start"], norm(clipped), places=14)
        self.assertAlmostEqual(values["low_rank_correction_norm_start"], norm(residuals), places=14)
        self.assertAlmostEqual(values["correction_to_gradient_ratio_start"],
                               norm(residuals) / (norm(clipped) + 1e-12), places=13)
        self.assertEqual(values["encoder_grad_norm_start"], values["encoder_grad_norm_raw_start"])

    def test_raw_measurements_match_fo_proto_maml_reference(self):
        # Supply its baseline-specific helper, without importing a second model.py.
        spec = importlib.util.spec_from_file_location("fo_reference_helpers", ROOT.parent /
                                                      "fo_proto_maml/helpers_fo_proto_maml.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        spec = importlib.util.spec_from_file_location("fo_reference_diagnostic", ROOT.parent /
                                                      "fo_proto_maml/eval_inner_steps.py")
        reference = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, helpers_fo_proto_maml=helper):
            spec.loader.exec_module(reference)
        self.assertEqual(reference.TEST_EPISODES, diagnostic.TEST_EPISODES)
        predictor = self.fit()
        expected = reference.support_diagnostics(self.learner, *self.task.support_set[:2],
                                                 self.task.num_ways, predictor.weights)
        actual = self.diagnostics(predictor.weights)
        self.assertEqual(tuple(actual[k] for k in ("support_loss_start", "encoder_grad_norm_start",
                                                   "support_loss_after5")), expected)

    def test_real_resnet_nonzero_conditioned_corrections_and_prediction_parity(self):
        learner = self.learner
        learner.learner = baseline.make_encoder(learner.model_args).eval()
        learner.transport = TaskConditionedTransport(learner.learner, learner.config).eval()
        with torch.no_grad():
            # Keep synthetic prototype logits away from float32 CE saturation.
            for name, parameter in learner.learner.named_parameters():
                if name.endswith("weight") and parameter.ndim == 1:
                    parameter.fill_(.1)
            for u in learner.transport.u.values():
                u.normal_(0, .01)
            learner.transport.gate_net.out.weight.normal_(0, .01)
            learner.transport.gate_net.out.bias.normal_(0, .01)
        x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1])
        task = SimpleNamespace(dataset="synthetic_images", num_ways=2, num_shots=2,
                               support_set=(x, y, y), query_set=(x + .1, y, y))
        normal = self.fit(task).predict(task.query_set[0])
        before = copy.deepcopy((learner.learner.state_dict(), learner.transport.state_dict()))
        row = diagnostic.evaluate_task(learner, task, 1)
        self.assertGreater(row["low_rank_correction_norm_start"], 0)
        self.assertEqual(row["acc_steps5"], float((normal.argmax(1) == y.numpy()).mean()))
        np.testing.assert_array_equal(self.fit(task).predict(task.query_set[0]), normal)
        for saved, module in zip(before, (learner.learner, learner.transport)):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, saved[key], rtol=0, atol=0)

    def test_zero_steps_is_pure_prototype_and_five_steps_predictions_are_exact(self):
        normal = self.fit().predict(self.task.query_set[0])
        original_fit = self.learner.fit
        predictions, steps = [], []

        def observe(support):
            s = self.learner.config["method_config"]["inner_steps"]
            steps.append(s)
            predictor = original_fit(support)
            predictions.append(predictor.predict(self.task.query_set[0]))
            if s == 0:
                body = list(self.learner.learner.parameters())
                features = self.learner.learner.forward_weights(self.task.support_set[0], body, embedding=True)
                head = diagnostic.prototype_head(features, self.task.support_set[1], self.task.num_ways)
                for actual, expected in zip(predictor.weights, body + list(head)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return predictor

        with patch.object(self.learner, "fit", side_effect=observe):
            row = diagnostic.evaluate_task(self.learner, self.task, 1)
        self.assertEqual(steps, [0, 5])
        np.testing.assert_array_equal(predictions[1], normal)
        np.testing.assert_array_equal(self.fit().predict(self.task.query_set[0]), normal)
        self.assertEqual(row["adaptation_gain"], row["acc_steps5"] - row["acc_steps0"])

    def test_diagnostics_preserve_state_gradients_metrics_rng_and_optimizer(self):
        predictor = self.fit()
        params = list(self.learner.learner.parameters()) + list(self.learner.transport.parameters())
        for p in params:
            p.grad = torch.ones_like(p)
        optimizer = torch.optim.Adam(params)
        optimizer.step()
        before = copy.deepcopy((self.learner.learner.state_dict(), self.learner.transport.state_dict(),
                                self.learner.state, optimizer.state_dict()))
        grads = [p.grad.clone() for p in params]
        rng = torch.random.get_rng_state().clone(), random.getstate(), np.random.get_state()
        with patch.object(self.learner.transport, "_record_conditioning", side_effect=AssertionError), \
                patch.object(self.learner.transport, "transport_gradient", side_effect=AssertionError):
            self.diagnostics(predictor.weights)
        after = (self.learner.learner.state_dict(), self.learner.transport.state_dict(),
                 self.learner.state, optimizer.state_dict())

        def check(a, b):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            elif isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    check(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    check(x, y)
            else:
                self.assertEqual(a, b)
        check(before, after)
        check(grads, [p.grad for p in params])
        check(rng[:2], (torch.random.get_rng_state(), random.getstate()))
        np.testing.assert_equal(rng[2], np.random.get_state())
        self.assertEqual(self.learner.transport._count, 0)
        self.assertEqual(self.learner.transport._task_count, 0)

    def test_config_restored_on_success_fit_predict_and_diagnostic_failure(self):
        original = self.learner.config["method_config"]
        diagnostic.evaluate_task(self.learner, self.task, 1)
        self.assertIs(self.learner.config["method_config"], original)
        for owner, method in ((self.learner, "fit"), (baseline.MyPredictor, "predict"),
                              (diagnostic, "support_diagnostics")):
            with self.subTest(method=method), patch.object(owner, method, side_effect=RuntimeError("test")):
                with self.assertRaisesRegex(RuntimeError, "test"):
                    diagnostic.evaluate_task(self.learner, self.task, 1)
            self.assertIs(self.learner.config["method_config"], original)

    def test_no_clip_and_zero_residual_and_finite_checks(self):
        self.learner.config["method_config"]["grad_clip"] = None
        with torch.no_grad():
            for u in self.learner.transport.u.values():
                u.zero_()
        row = diagnostic.evaluate_task(self.learner, self.task, 1)
        self.assertEqual(row["encoder_grad_norm_raw_start"], row["encoder_grad_norm_clipped_start"])
        self.assertEqual(row["low_rank_correction_norm_start"], 0)
        self.assertEqual(row["correction_to_gradient_ratio_start"], 0)
        with patch.object(diagnostic.F, "cross_entropy", return_value=torch.tensor(float("nan"), requires_grad=True)):
            with self.assertRaises((ValueError, RuntimeError)):
                diagnostic.evaluate_task(self.learner, self.task, 1)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                diagnostic.require_finite({"support_loss_start": value})

    def test_state_guard_detects_mutation_and_rejects_training(self):
        with self.assertRaisesRegex(RuntimeError, "buffer"):
            with diagnostic.observational_diagnostics(self.learner):
                self.learner.learner.diagnostic_test_buffer.add_(1)
        self.learner.transport.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            self.diagnostics([])

    def test_checkpoint_validation_rejects_every_wrong_or_missing_setting(self):
        diagnostic.validate_checkpoint_config(self.learner.config)
        for section, key, value in ((None, "method", "fo-proto-maml"),
                ("lrsg", "enabled", False), ("task_conditioning", "enabled", False),
                ("task_conditioning", "scalar_delta_scale", 1.0),
                ("task_conditioning", "low_rank_delta_scale", 0.0)):
            for missing in (False, True):
                config = copy.deepcopy(self.learner.config)
                target = config if section is None else config[section]
                if missing:
                    del target[key]
                else:
                    target[key] = value
                with self.subTest(section=section, key=key, missing=missing):
                    with self.assertRaisesRegex(ValueError, "LR-only-TC checkpoint requires"):
                        diagnostic.validate_checkpoint_config(config)

    def write_reference(self, path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("task_id", "dataset", "num_ways", "num_shots", "accuracy"))
            writer.writeheader()
            writer.writerows(rows)

    def reference_rows(self, tasks):
        return [dict(task_id=i, dataset=t.dataset, num_ways=t.num_ways, num_shots=t.num_shots,
                     accuracy=float((self.fit(t).predict(t.query_set[0]).argmax(1) == t.query_set[1].numpy()).mean()))
                for i, t in enumerate(tasks, 1)]

    def test_parity_first_mismatch_fields_count_order_and_serialization(self):
        row = diagnostic.evaluate_task(self.learner, self.task, 1)
        original = self.reference_rows([self.task])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_results.csv"
            for field, bad in (("dataset", "WRONG"), ("num_ways", 99), ("num_shots", 99),
                               ("task_id", 2), ("accuracy", .123456789), ("accuracy", "nan"),
                               ("accuracy", "")):
                rows = copy.deepcopy(original)
                rows[0][field] = bad
                self.write_reference(path, rows)
                reference = diagnostic.ReferenceTaskResults(path, 1)
                with self.subTest(field=field, bad=bad), self.assertRaisesRegex(ValueError, "task 1"):
                    reference.check(row)
            self.write_reference(path, original)
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                diagnostic.ReferenceTaskResults(path, 2)
            reference = diagnostic.ReferenceTaskResults(path, 1)
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                reference.finish()
            reference.check(dict(row, acc_steps5=row["acc_steps5"] + 1e-14))
            reference.finish()
            with self.assertRaisesRegex(ValueError, "extra task"):
                reference.check(row)
            path.write_text("task_id,dataset\n1,DOG\n")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                diagnostic.ReferenceTaskResults(path, 1)

    def test_saved_config_normal_load_cli_and_deterministic_tasks_seeds_93_94_95(self):
        # Exercise real save/load and real CompetitionDataLoader with synthetic data.
        # Only the heavyweight ResNet and on-disk dataset creation are substituted.
        def tiny_encoder(args):
            model = Tiny().double()
            model.register_buffer("diagnostic_test_buffer", torch.tensor(2.))
            return model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.learner.save(root)
            for seed in (93, 94, 95):
                tasks = list(self.tasks(seed))
                again = list(self.tasks(seed))
                for a, b in zip(tasks, again):
                    self.assertEqual((a.dataset, a.num_ways, a.num_shots), (b.dataset, b.num_ways, b.num_shots))
                    for x, y in zip(a.support_set + a.query_set, b.support_set + b.query_set):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
                reference = root / f"reference_{seed}.csv"
                self.write_reference(reference, self.reference_rows(tasks))
                out = root / f"diagnostic_{seed}.csv"
                stdout = io.StringIO()
                with patch.object(baseline, "make_encoder", side_effect=tiny_encoder), \
                        patch.object(baseline, "read_config", side_effect=AssertionError("Repository config read")), \
                        patch.object(diagnostic, "prepare_datasets_information", return_value=(None, None, {"test": 1})) as prepare, \
                        patch.object(diagnostic, "create_datasets", return_value=self.datasets), redirect_stdout(stdout):
                    diagnostic.main(["--checkpoint", str(root / "max-va.pth"), "--input_data_dir", "synthetic",
                                     "--out_csv", str(out), "--seed", str(seed), "--test_tasks_per_dataset", "2",
                                     "--reference_task_results", str(reference)])
                prepare.assert_called_once_with("synthetic", self.learner.config["validation_datasets"], seed, False)
                self.assertIn("Reference parity PASSED: 4 tasks", stdout.getvalue())
                with out.open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 4)
                self.assertEqual(tuple(rows[0]), diagnostic.FIELDS)
                self.assertTrue(all(np.isfinite(float(row[key])) for row in rows for key in diagnostic.METRICS))
            self.assertFalse(torch.equal(list(self.tasks(93))[0].support_set[0],
                                         list(self.tasks(94))[0].support_set[0]))


if __name__ == "__main__":
    unittest.main()
