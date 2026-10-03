"""CPU synthetic checks for the gamma sweep; not a trained Meta-Album experiment."""
import copy
import csv
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_eval_condition_controls import Tiny  # noqa: E402
from test_eval_inner_steps import SyntheticDataset  # noqa: E402
import model as baseline  # noqa: E402
import eval_beta_sweep as sweep  # noqa: E402
from task_transport import TaskConditionedTransport  # noqa: E402
from cdmetadl.helpers.scoring_helpers import accuracy, read_results_file  # noqa: E402


class BetaSweepTests(unittest.TestCase):
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
        config["lrsg"]["beta"] = 0.75  # gamma must multiply the SAVED beta, not replace it
        config["method_config"].update(inner_steps=5, encoder_lr=.07,
                                       classifier_lr=.13, grad_clip=.005, eval_gamma=1.0)
        model = Tiny().double().eval()
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
        return sweep.CompetitionDataLoader(self.datasets, sweep.TEST_EPISODES,
                                           seed, test_generator=True).generator(count)

    def weights(self, task=None):
        task = self.task if task is None else task
        return self.learner.fit((*task.support_set, task.num_ways, task.num_shots)).weights

    def sweep_weights(self, gammas):
        """Adapted weights seen by evaluate_task for steps0 and each gamma, in call order."""
        original_fit, seen = self.learner.fit, []

        def observe(support):
            predictor = original_fit(support)
            seen.append((self.learner.config["method_config"]["inner_steps"],
                         self.learner.transport.beta,
                         [w.detach().clone() for w in predictor.weights]))
            return predictor

        with patch.object(self.learner, "fit", side_effect=observe):
            row = sweep.evaluate_task(self.learner, self.task, 1, gammas)
        return row, seen

    def assert_same(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_gamma_one_is_the_trained_model_and_beta_config_are_restored(self):
        normal = self.weights()
        original_config = self.learner.config["method_config"]
        row, seen = self.sweep_weights((0.0, 1.0, 2.0))
        self.assertEqual([(s, b) for s, b, _ in seen],
                         [(0, .75), (5, 0.0), (5, .75), (5, 1.5)])
        self.assert_same(seen[2][2], normal)
        self.assertEqual(self.learner.transport.beta, .75)
        self.assertIs(self.learner.config["method_config"], original_config)
        self.assert_same(self.weights(), normal)
        self.assertEqual(tuple(row), sweep.ID_FIELDS + ("acc_steps0", "acc_gamma_0",
                                                         "acc_gamma_1", "acc_gamma_2"))

    def test_gamma_zero_keeps_encoder_and_scalar_gates_and_removes_only_the_residual(self):
        _, seen = self.sweep_weights((0.0, 1.0))
        saved_u = copy.deepcopy(self.learner.transport.u.state_dict())
        with torch.no_grad():
            for u in self.learner.transport.u.values():
                u.zero_()
        without_residual = self.weights()
        self.learner.transport.u.load_state_dict(saved_u)
        self.assert_same(seen[1][2], without_residual)
        # It still adapts (scalar-gated gradient steps), so it is not the prototype start,
        # and the residual is active in the trained model.
        self.assertFalse(torch.equal(seen[1][2][0], seen[0][2][0]))
        self.assertFalse(torch.equal(seen[1][2][0], seen[2][2][0]))

    def test_gamma_scales_the_whole_residual_not_delta_c(self):
        _, seen = self.sweep_weights((1.0, 2.0))
        with torch.no_grad():
            for u in self.learner.transport.u.values():
                u.mul_(2)  # the residual is linear in U: 2 * beta * U diag(1+dc) V^T G
        doubled_u = self.weights()
        with torch.no_grad():
            for u in self.learner.transport.u.values():
                u.div_(2)
        for a, b in zip(seen[2][2], doubled_u):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
        original_condition = self.learner.transport.condition

        def doubled_delta(embedding):
            delta_a, delta_c = original_condition(embedding)
            return delta_a, {key: 2 * value for key, value in delta_c.items()}

        with patch.object(self.learner.transport, "condition", side_effect=doubled_delta):
            doubled_delta_c = self.weights()
        self.assertFalse(torch.allclose(seen[2][2][0], doubled_delta_c[0], rtol=1e-6, atol=1e-9))

    def test_beta_restored_when_fit_or_predict_fails(self):
        for owner, method in ((self.learner, "fit"), (baseline.MyPredictor, "predict")):
            original = self.learner.config["method_config"]
            implementation, calls = getattr(owner, method), []

            def fail_during_gamma(*args, **kwargs):
                calls.append(self.learner.transport.beta)
                if len(calls) == 2:
                    raise RuntimeError("test")
                return implementation(*args, **kwargs)

            with self.subTest(method=method), patch.object(owner, method, autospec=True,
                                                           side_effect=fail_during_gamma):
                with self.assertRaisesRegex(RuntimeError, "test"):
                    sweep.evaluate_task(self.learner, self.task, 1, (2.0, 1.0))
            self.assertEqual(calls, [.75, 1.5])
            self.assertEqual(self.learner.transport.beta, .75)
            self.assertIs(self.learner.config["method_config"], original)
        with self.assertRaisesRegex(RuntimeError, "inside"):
            with sweep.scaled_low_rank(self.learner.transport, 4.0):
                self.assertEqual(self.learner.transport.beta, 3.0)
                raise RuntimeError("inside")
        self.assertEqual(self.learner.transport.beta, .75)

    def test_scorer_parity_after_six_decimal_serialization_near_tie(self):
        # Raw argmax picks class 1; ingestion serialization produces a tie and
        # the real scorer picks class 0. Original tests missed this difference.
        probabilities = np.array([[.49999996, .50000004], [.1, .9]])
        truth = torch.tensor([1, 1])
        task = SimpleNamespace(support_set=self.task.support_set,
                               query_set=(self.task.query_set[0], truth, None),
                               num_ways=2, num_shots=1, dataset="TIE")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_1.predict"
            np.savetxt(path, probabilities, fmt="%f")
            expected = accuracy(truth.numpy(), read_results_file(str(path)))
        self.assertEqual(expected, .5)
        self.assertEqual(float((probabilities.argmax(1) == truth.numpy()).mean()), 1.)
        predictor = SimpleNamespace(predict=lambda _: probabilities)
        with patch.object(self.learner, "fit", return_value=predictor):
            row = sweep.evaluate_task(self.learner, task, 1, (0., 1., 2.))
        for column in ("acc_steps0", "acc_gamma_0", "acc_gamma_1", "acc_gamma_2"):
            self.assertEqual(row[column], expected)

    def test_same_tensors_gamma_order_and_state_rng_are_preserved(self):
        before = copy.deepcopy((self.learner.learner.state_dict(),
                                self.learner.transport.state_dict()))
        parameters = list(self.learner.learner.parameters()) + list(self.learner.transport.parameters())
        for p in parameters:
            p.grad = torch.ones_like(p)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        original_fit, original_predict = self.learner.fit, baseline.MyPredictor.predict
        supports, queries = [], []

        def observe_fit(support):
            supports.append(support)
            return original_fit(support)

        def observe_predict(predictor, query):
            queries.append(query)
            probabilities = original_predict(predictor, query)
            # Even an incidental RNG draw must not escape the diagnostic.
            random.random()
            np.random.rand()
            torch.rand(1)
            return probabilities

        with patch.object(self.learner, "fit", side_effect=observe_fit), \
                patch.object(baseline.MyPredictor, "predict", autospec=True, side_effect=observe_predict):
            forward = sweep.evaluate_task(self.learner, self.task, 1, (0., .5, 1., 2.))
            reverse = sweep.evaluate_task(self.learner, self.task, 1, (2., 1., .5, 0.))
        self.assertEqual(forward, reverse)
        for support in supports:
            self.assertIs(support[0], self.task.support_set[0])
            self.assertIs(support[1], self.task.support_set[1])
        for query in queries:
            self.assertIs(query, self.task.query_set[0])
        self.assertEqual(random.getstate(), python_rng)
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_rng[0])
        np.testing.assert_array_equal(current_numpy[1], numpy_rng[1])
        self.assertEqual(current_numpy[2:], numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        for saved, module in zip(before, (self.learner.learner, self.learner.transport)):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
        for p in parameters:
            torch.testing.assert_close(p.grad, torch.ones_like(p), rtol=0, atol=0)

    def test_rejects_training_mode_and_nonfinite_predictions_restores_beta(self):
        self.learner.transport.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            sweep.evaluate_task(self.learner, self.task, 1, (1.,))
        self.learner.transport.eval()
        original = baseline.MyPredictor.predict
        calls = []

        def nonfinite_at_gamma(predictor, query):
            calls.append(self.learner.transport.beta)
            if len(calls) == 2:
                return np.full((len(query), self.task.num_ways), np.nan)
            return original(predictor, query)

        with patch.object(baseline.MyPredictor, "predict", autospec=True, side_effect=nonfinite_at_gamma):
            with self.assertRaisesRegex(ValueError, "nonfinite predictions at gamma=2"):
                sweep.evaluate_task(self.learner, self.task, 1, (2., 1.))
        self.assertEqual(self.learner.transport.beta, .75)
        with self.assertRaisesRegex(ValueError, "finite"):
            with sweep.scaled_low_rank(self.learner.transport, float("inf")):
                self.fail("nonfinite gamma accepted")
        self.assertEqual(self.learner.transport.beta, .75)

    def test_real_resnet_functional_batchnorm_and_gamma_two_reload_parity(self):
        config = copy.deepcopy(self.learner.config)
        config["method_config"]["inner_steps"] = 1
        args = dict(num_classes=2, dev="cpu", num_blocks=18, pretrained=False)
        encoder = baseline.make_encoder(args).eval()
        transport = TaskConditionedTransport(encoder, config).eval()
        with torch.no_grad():
            for u in transport.u.values():
                u.normal_(0, .01)
        config["method_config"]["eval_gamma"] = 2.0
        learner = baseline.MyLearner(args, baseline.snapshot(encoder, transport), config, .5)
        task = SimpleNamespace(support_set=(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]), None),
                               query_set=(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]), None),
                               num_ways=2, num_shots=1, dataset="SYNTHETIC_RESNET")
        before = copy.deepcopy(learner.learner.state_dict())
        normal = learner.fit((*task.support_set, 2, 1))
        probabilities = normal.predict(task.query_set[0])
        with patch.object(learner, "fit", wraps=learner.fit) as fit:
            row = sweep.evaluate_task(learner, task, 1, (0., 1., 2.))
        self.assertEqual(fit.call_count, 4)
        self.assertEqual(row["acc_gamma_2"], sweep.scorer_accuracy(probabilities, task.query_set[1].numpy()))
        # Reusing the loaded encoder is equivalent to ingestion's fresh load.
        with tempfile.TemporaryDirectory() as directory:
            learner.save(directory)
            reloaded = baseline.MyLearner()
            reloaded.load(directory)
            fresh = reloaded.fit((*task.support_set, 2, 1)).predict(task.query_set[0])
        np.testing.assert_array_equal(fresh, probabilities)
        for key, value in learner.learner.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_output_preserves_existing_files_and_removes_failed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "max-va.pth"
            path.write_bytes(b"checkpoint sentinel")
            arguments = ["--checkpoint", str(path), "--input_data_dir", "synthetic", "--out_csv", str(path)]
            with patch.object(sweep.MyLearner, "load") as load, \
                    patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                sweep.main(arguments)
            load.assert_not_called()
            self.assertEqual(path.read_bytes(), b"checkpoint sentinel")
            # Exclusive open remains safe even if a file appears after parsing.
            with self.assertRaises(FileExistsError):
                with sweep.new_output_csv(path):
                    self.fail("existing file opened")
            self.assertEqual(path.read_bytes(), b"checkpoint sentinel")
            out = Path(directory) / "sweep.csv"
            with self.assertRaisesRegex(RuntimeError, "partial"):
                with sweep.new_output_csv(out) as handle:
                    handle.write("partial result")
                    raise RuntimeError("partial")
            self.assertFalse(out.exists())

    def test_gamma_parsing(self):
        self.assertEqual(sweep.parse_gammas(sweep.DEFAULT_GAMMAS), (0.0, .25, .5, 1.0, 2.0, 4.0))
        self.assertEqual([sweep.gamma_field(g) for g in (0.0, .25, 1.0, 4.0)],
                         ["acc_gamma_0", "acc_gamma_0.25", "acc_gamma_1", "acc_gamma_4"])
        for bad in ("0,0.5,2", "1,-1", "1,1.0", "1,nan", "1,inf", "1,x", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sweep.parse_gammas(bad)
        self.assertEqual([sweep.shot_bin(s) for s in (1, 2, 3, 5, 6, 10, 11, 20)],
                         ["1", "2", "3-5", "3-5", "6-10", "6-10", "11-20", "11-20"])
        with self.assertRaises(ValueError):
            sweep.shot_bin(21)

    def reference_rows(self, tasks):
        rows = []
        for i, t in enumerate(tasks, 1):
            predictor = self.learner.fit((*t.support_set, t.num_ways, t.num_shots))
            # Exercise the actual file-based scorer contract, not the sweep's
            # own helper, so a raw-argmax regression cannot pass this reference.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "task.predict"
                np.savetxt(path, predictor.predict(t.query_set[0]), fmt="%f")
                scored = accuracy(t.query_set[1].numpy(), read_results_file(str(path)))
            rows.append(dict(task_id=i, dataset=t.dataset, num_ways=t.num_ways,
                             num_shots=t.num_shots, accuracy=scored))
        return rows

    def test_cli_saved_checkpoint_same_tasks_parity_and_csv(self):
        def tiny_encoder(args):
            return Tiny().double()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.learner.save(root)
            before = copy.deepcopy((self.learner.learner.state_dict(), self.learner.transport.state_dict()))
            tasks = list(self.tasks(93))
            reference = root / "task_results.csv"
            rows = self.reference_rows(tasks)
            with reference.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            out, stdout = root / "sweep.csv", io.StringIO()
            patches = (patch.object(baseline, "make_encoder", side_effect=tiny_encoder),
                       patch.object(sweep, "prepare_datasets_information", return_value=(None, None, {"test": 1})),
                       patch.object(sweep, "create_datasets", return_value=self.datasets))
            arguments = ["--checkpoint", str(root / "max-va.pth"), "--input_data_dir", "synthetic",
                         "--out_csv", str(out), "--seed", "93", "--test_tasks_per_dataset", "2",
                         "--reference_task_results", str(reference)]
            with patches[0], patches[1] as prepare, patches[2], redirect_stdout(stdout):
                sweep.main(arguments + ["--gammas", "0,0.5,1,4"])
            prepare.assert_called_once_with("synthetic", self.learner.config["validation_datasets"], 93, False)
            text = stdout.getvalue()
            self.assertIn("saved beta=0.75", text)
            self.assertIn("Reference parity PASSED (gamma=1): 4 tasks", text)
            self.assertIn("Exploratory only", text)
            with out.open() as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(tuple(written[0]), sweep.ID_FIELDS + (
                "acc_steps0", "acc_gamma_0", "acc_gamma_0.5", "acc_gamma_1", "acc_gamma_4"))
            self.assertEqual(len(written), 4)
            for row, expected, task in zip(written, rows, tasks):
                self.assertEqual(float(row["acc_gamma_1"]), expected["accuracy"])
                self.assertEqual((row["dataset"], int(row["num_ways"]), int(row["num_shots"])),
                                 (task.dataset, task.num_ways, task.num_shots))
            # A wrong reference must stop the sweep at the first task.
            rows[0]["accuracy"] = rows[0]["accuracy"] + .05
            with reference.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            failed_out = root / "failed_sweep.csv"
            with patches[0], patches[1], patches[2], redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Reference parity mismatch at task 1"):
                    sweep.main(arguments + ["--out_csv", str(failed_out)])
            self.assertFalse(failed_out.exists())
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                    patch.object(sys, "stderr", io.StringIO()):
                sweep.main(arguments + ["--gammas", "0,2"])
            for saved, module in zip(before, (self.learner.learner, self.learner.transport)):
                for key, value in module.state_dict().items():
                    torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
