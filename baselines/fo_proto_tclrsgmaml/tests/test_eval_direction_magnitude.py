"""CPU synthetic checks for the direction x magnitude decomposition; not a trained experiment."""
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
import eval_direction_magnitude as decomposition  # noqa: E402
from helpers_fo_proto_tclrsgmaml import adapt  # noqa: E402
from task_transport import TaskConditionedTransport  # noqa: E402
from cdmetadl.helpers.scoring_helpers import accuracy, read_results_file  # noqa: E402


def norm(tensors):
    return decomposition.squared_norm(tensors).sqrt().item()


def cosine(a, b):
    dot = sum((x.double() * y.double()).sum() for x, y in zip(a, b)).item()
    return dot / (norm(a) * norm(b))


class DirectionMagnitudeTests(unittest.TestCase):
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
        config["lrsg"]["beta"] = 0.75
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
        return decomposition.CompetitionDataLoader(
            self.datasets, decomposition.TEST_EPISODES, seed, test_generator=True).generator(count)

    def variant(self, name, gamma=1.0, steps=None, task=None):
        task = self.task if task is None else task
        fast, statistics = decomposition.adapt_variant(
            self.learner, task.support_set[0], task.support_set[1], task.num_ways,
            name, gamma, steps)
        return fast, statistics

    def assert_same(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b.detach(), rtol=0, atol=0)

    def test_full_is_the_trained_model_and_plain_is_the_untransported_update(self):
        support = (*self.task.support_set, self.task.num_ways, self.task.num_shots)
        for gamma in (1.0, 2.0):
            self.learner.config["method_config"]["eval_gamma"] = gamma
            self.assert_same(self.variant("full", gamma)[0], self.learner.fit(support).weights)
        self.learner.config["method_config"]["eval_gamma"] = 1.0
        model = self.learner.learner
        plain = adapt(model, list(model.parameters()), self.task.support_set[0],
                      self.task.support_set[1], self.learner.config["method_config"],
                      self.task.num_ways, None, phase="test")
        self.assert_same(self.variant("plain")[0], plain)
        self.assertFalse(torch.equal(self.variant("plain")[0][0], self.variant("full")[0][0]))
        # steps=0 is the prototype start of the checkpoint encoder.
        start = self.variant("plain", steps=0)[0]
        self.assert_same(start[:2], list(model.parameters()))
        self.assertEqual(decomposition.self_check(
            self.learner, self.task.support_set[0], self.task.support_set[1],
            self.task.num_ways, 1.0), 0.0)

    def test_variant_gradients_swap_only_norm_or_only_direction(self):
        generator = torch.Generator().manual_seed(3)
        raw = [torch.randn(4, 3, generator=generator, dtype=torch.float64),
               torch.randn(4, generator=generator, dtype=torch.float64)]
        full = [7 * torch.randn(4, 3, generator=generator, dtype=torch.float64),
                .2 * torch.randn(4, generator=generator, dtype=torch.float64)]
        self.assert_same(decomposition.variant_gradients("plain", raw, full), raw)
        self.assert_same(decomposition.variant_gradients("full", raw, full), full)
        for kind, direction, magnitude in (("mag", raw, full), ("dir", full, raw)):
            layer = decomposition.variant_gradients(f"{kind}_layer", raw, full)
            everything = decomposition.variant_gradients(f"{kind}_global", raw, full)
            for out, d, m in zip(layer, direction, magnitude):
                self.assertAlmostEqual(norm([out]), norm([m]), places=12)
                self.assertAlmostEqual(cosine([out], [d]), 1.0, places=12)
            self.assertAlmostEqual(norm(everything), norm(magnitude), places=12)
            self.assertAlmostEqual(cosine(everything, direction), 1.0, places=12)
            # One scalar keeps the direction's own allocation across tensors.
            self.assertAlmostEqual(norm([everything[0]]) / norm([everything[1]]),
                                   norm([direction[0]]) / norm([direction[1]]), places=10)
            self.assertNotAlmostEqual(norm([everything[0]]), norm([magnitude[0]]), places=3)
        zero = [torch.zeros(4, 3, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)]
        for name in decomposition.VARIANTS:
            for out in decomposition.variant_gradients(name, zero, zero):
                self.assertEqual(out.abs().sum().item(), 0.0)
        for out in decomposition.variant_gradients("dir_layer", raw, zero):
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(out.abs().sum().item(), 0.0)
        with self.assertRaisesRegex(ValueError, "Unknown variant"):
            decomposition.variant_gradients("mag_tensor", raw, full)

    def test_one_step_updates_have_the_swapped_norms_and_share_the_head(self):
        rate = self.learner.config["method_config"]["encoder_lr"]
        start = self.variant("plain", steps=0)[0]
        steps = {name: self.variant(name, steps=1)[0] for name in decomposition.VARIANTS}
        update = {name: [(s - w) / rate for s, w in zip(start[:2], fast[:2])]
                  for name, fast in steps.items()}
        g, pg = update["plain"], update["full"]
        for name, direction, magnitude in (("mag", g, pg), ("dir", pg, g)):
            for out, d, m in zip(update[f"{name}_layer"], direction, magnitude):
                self.assertAlmostEqual(norm([out]) / norm([m]), 1.0, places=9)
                self.assertAlmostEqual(cosine([out], [d]), 1.0, places=9)
            self.assertAlmostEqual(norm(update[f"{name}_global"]) / norm(magnitude), 1.0, places=9)
            self.assertAlmostEqual(cosine(update[f"{name}_global"], direction), 1.0, places=9)
        # The low-rank residual changes the direction of the ranked tensor only.
        self.assertLess(cosine([g[0]], [pg[0]]), 1 - 1e-6)
        self.assertAlmostEqual(cosine([g[1]], [pg[1]]), 1.0, places=12)
        for name in decomposition.VARIANTS:
            self.assert_same(steps[name][2:], steps["plain"][2:])
        statistics = self.variant("full", steps=1)[1]
        self.assertAlmostEqual(statistics["norm_ratio_step1"], norm(pg) / norm(g), places=9)
        self.assertAlmostEqual(statistics["cos_g_pg_step1"], cosine(g, pg), places=9)
        self.assertGreater(statistics["support_loss_start"], 0.0)
        self.assertEqual(statistics["zero_gradient_step1"], 0)
        self.assertEqual(tuple(statistics), decomposition.STEP1_FIELDS)
        self.assertIsNone(self.variant("plain", steps=1)[1])

    def test_zero_first_step_gradient_is_flagged_and_left_out_of_the_means(self):
        loss = torch.tensor(0.25, dtype=torch.float64)
        zero = [torch.zeros(4, 3, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)]
        some = [torch.ones(4, 3, dtype=torch.float64), torch.ones(4, dtype=torch.float64)]
        undefined = decomposition.first_step_statistics(loss, zero, zero)
        self.assertEqual(undefined, dict(support_loss_start=.25, zero_gradient_step1=1,
                                         norm_ratio_step1=None, cos_g_pg_step1=None))
        # A nonzero gradient that is transported to zero has ratio 0 but no direction.
        self.assertEqual(decomposition.first_step_statistics(loss, some, zero),
                         dict(support_loss_start=.25, zero_gradient_step1=0,
                              norm_ratio_step1=0.0, cos_g_pg_step1=None))
        defined = decomposition.first_step_statistics(loss, some, [3 * g for g in some])
        self.assertEqual(defined["zero_gradient_step1"], 0)
        self.assertAlmostEqual(defined["norm_ratio_step1"], 3.0, places=12)
        self.assertAlmostEqual(defined["cos_g_pg_step1"], 1.0, places=12)

        # grad_clip=0 zeroes every update gradient: nothing adapts, nothing is defined.
        self.learner.config["method_config"]["grad_clip"] = 0
        row = decomposition.evaluate_task(self.learner, self.task, 1, 1.0, check=True)
        row.pop("_self_check_max_abs_diff")
        self.assertEqual((row["zero_gradient_step1"], row["norm_ratio_step1"],
                          row["cos_g_pg_step1"]), (1, None, None))
        self.assertGreater(row["support_loss_start"], 0.0)
        for name in decomposition.VARIANTS:
            self.assertEqual(row[f"acc_{name}"], row["acc_steps0"])
        with io.StringIO() as handle:
            writer = csv.DictWriter(handle, fieldnames=decomposition.FIELDS)
            writer.writeheader()
            writer.writerow(row)
            written = next(csv.DictReader(io.StringIO(handle.getvalue())))
        self.assertEqual((written["zero_gradient_step1"], written["norm_ratio_step1"],
                          written["cos_g_pg_step1"]), ("1", "", ""))

        # Means: one undefined task must not pull the ratio/cosine towards zero.
        valid = dict(row, zero_gradient_step1=0, norm_ratio_step1=200.0, cos_g_pg_step1=.02)
        sums = decomposition.defaultdict(lambda: decomposition.defaultdict(float))
        decomposition.accumulate(sums, ("ALL", "1"), row)
        decomposition.accumulate(sums, ("ALL", "11-20"), valid)
        self.assertEqual((sums["ALL"]["n"], sums["ALL"]["n_acc_full"],
                          sums["ALL"]["n_norm_ratio_step1"], sums["ALL"]["n_cos_g_pg_step1"]),
                         (2, 2, 1, 1))
        self.assertEqual(sums["ALL"]["norm_ratio_step1"], 200.0)
        self.assertEqual(sums["ALL"]["zero_gradient_step1"], 1)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            decomposition.print_summary(sums)
        tables = stdout.getvalue().split("Shared first step.")[1].splitlines()
        cells = {line.split()[0]: line.split()[1:] for line in tables[2:6] if line.strip()}
        loss_text = f"{row['support_loss_start']:.4g}"
        self.assertEqual(cells["ALL"], [loss_text, "50.00", "200", "0.02", "1"])
        self.assertEqual(cells["1"], [loss_text, "100.00", "n/a", "n/a", "0"])
        self.assertEqual(cells["11-20"], [loss_text, "0.00", "200", "0.02", "1"])

    def test_row_fields_and_gamma_reach_only_the_transported_variants(self):
        row = decomposition.evaluate_task(self.learner, self.task, 1, 1.0, check=True)
        self.assertEqual(row.pop("_self_check_max_abs_diff"), 0.0)
        self.assertEqual(tuple(row), decomposition.FIELDS)
        for name in decomposition.COUNTERFACTUALS:
            self.assertEqual(row[f"diverged_{name}"], 0)
        self.assert_same(self.variant("plain", 1.0)[0], self.variant("plain", 4.0)[0])
        for name in ("mag_layer", "mag_global", "full"):
            self.assertFalse(torch.equal(self.variant(name, 1.0)[0][0],
                                         self.variant(name, 4.0)[0][0]))

    def test_counterfactual_divergence_is_chance_and_full_divergence_is_an_error(self):
        truth = self.task.query_set[1].numpy()
        uniform = np.full((len(truth), self.task.num_ways), 1.0 / self.task.num_ways)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.predict"
            np.savetxt(path, uniform, fmt="%f")
            chance = accuracy(truth, read_results_file(str(path)))
        original = baseline.MyPredictor.predict

        def nan_at(index):
            calls = []

            def predict(predictor, query):
                calls.append(1)
                if len(calls) == index:
                    return np.full((len(query), self.task.num_ways), np.nan)
                return original(predictor, query)
            return predict

        # Call order: steps0, then VARIANTS.
        with patch.object(baseline.MyPredictor, "predict", autospec=True, side_effect=nan_at(3)):
            row = decomposition.evaluate_task(self.learner, self.task, 1, 1.0)
        self.assertEqual(row["diverged_mag_layer"], 1)
        self.assertEqual(row["acc_mag_layer"], chance)
        self.assertEqual(sum(row[f] for f in decomposition.DIVERGED_FIELDS), 1)
        for index, label in ((1, "steps0"), (7, "full")):
            with patch.object(baseline.MyPredictor, "predict", autospec=True,
                              side_effect=nan_at(index)):
                with self.assertRaisesRegex(ValueError, f"nonfinite predictions at {label}"):
                    decomposition.evaluate_task(self.learner, self.task, 1, 1.0)
        # A nonfinite support loss stops the variant before prediction.
        real = decomposition.adapt_variant

        def lost(learner, support, labels, ways, variant, gamma, steps=None):
            if variant == "dir_global":
                return None, None
            return real(learner, support, labels, ways, variant, gamma, steps)

        with patch.object(decomposition, "adapt_variant", side_effect=lost):
            row = decomposition.evaluate_task(self.learner, self.task, 1, 1.0)
        self.assertEqual((row["diverged_dir_global"], row["acc_dir_global"]), (1, chance))
        self.learner.config["method_config"]["encoder_lr"] = float("nan")
        self.assertEqual(self.variant("plain")[0], None)

    def test_state_rng_and_config_are_preserved(self):
        before = copy.deepcopy((self.learner.learner.state_dict(),
                                self.learner.transport.state_dict()))
        parameters = list(self.learner.learner.parameters()) + list(self.learner.transport.parameters())
        for p in parameters:
            p.grad = torch.ones_like(p)
        config = self.learner.config["method_config"]
        saved_config, saved_beta = dict(config), self.learner.transport.beta
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        first = decomposition.evaluate_task(self.learner, self.task, 1, 1.0, check=True)
        second = decomposition.evaluate_task(self.learner, self.task, 1, 1.0, check=True)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), python_rng)
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_rng[0])
        np.testing.assert_array_equal(current_numpy[1], numpy_rng[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertIs(self.learner.config["method_config"], config)
        self.assertEqual(config, saved_config)
        self.assertEqual(self.learner.transport.beta, saved_beta)
        self.assertEqual(self.learner.transport._count, 0)
        for saved, module in zip(before, (self.learner.learner, self.learner.transport)):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
        for p in parameters:
            torch.testing.assert_close(p.grad, torch.ones_like(p), rtol=0, atol=0)

    def test_rejects_training_mode_zero_steps_and_a_broken_full_variant(self):
        self.learner.transport.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            decomposition.evaluate_task(self.learner, self.task, 1, 1.0)
        self.learner.transport.eval()
        real = decomposition.variant_gradients

        def broken(variant, raw, full):
            return [1.01 * g for g in real(variant, raw, full)] if variant == "full" else real(variant, raw, full)

        with patch.object(decomposition, "variant_gradients", side_effect=broken):
            with self.assertRaisesRegex(RuntimeError, "Self-check failed: full"):
                decomposition.evaluate_task(self.learner, self.task, 1, 1.0, check=True)
        self.learner.config["method_config"]["inner_steps"] = 0
        with self.assertRaisesRegex(ValueError, "inner_steps >= 1"):
            decomposition.evaluate_task(self.learner, self.task, 1, 1.0)

    def test_real_resnet_functional_batchnorm_parity(self):
        config = copy.deepcopy(self.learner.config)
        config["method_config"].update(inner_steps=2, grad_clip=10, encoder_lr=.01, classifier_lr=.01)
        args = dict(num_classes=2, dev="cpu", num_blocks=18, pretrained=False)
        encoder = baseline.make_encoder(args).eval()
        transport = TaskConditionedTransport(encoder, config).eval()
        with torch.no_grad():
            for u in transport.u.values():
                u.normal_(0, .01)
        learner = baseline.MyLearner(args, baseline.snapshot(encoder, transport), config, .5)
        # Random images are perfectly separated by the prototype head (support loss
        # exactly 0, no update at all). One image shared by both classes keeps the
        # support loss and the encoder gradient away from zero.
        support = torch.randn(6, 3, 32, 32)
        support[1] = support[0]
        task = SimpleNamespace(support_set=(support, torch.tensor([0, 1, 0, 1, 0, 1]), None),
                               query_set=(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]), None),
                               num_ways=2, num_shots=3, dataset="SYNTHETIC_RESNET")
        normal = learner.fit((*task.support_set, 2, 3))
        probabilities = normal.predict(task.query_set[0])
        row = decomposition.evaluate_task(learner, task, 1, 1.0, check=True)
        self.assertLess(row["_self_check_max_abs_diff"], 1e-6)
        self.assertEqual(row["acc_full"],
                         decomposition.scorer_accuracy(probabilities, task.query_set[1].numpy()))
        fast, statistics = decomposition.adapt_variant(
            learner, task.support_set[0], task.support_set[1], 2, "full", 1.0)
        for a, e in zip(fast, normal.weights):
            torch.testing.assert_close(a, e.detach(), rtol=0, atol=0)
        self.assertGreater(statistics["support_loss_start"], 0.0)
        self.assertEqual(statistics["zero_gradient_step1"], 0)
        self.assertGreater(statistics["norm_ratio_step1"], 0.0)
        self.assertLessEqual(abs(statistics["cos_g_pg_step1"]), 1.0 + 1e-9)
        # The realistic zero-gradient case: the prototype start separates distinct
        # random images perfectly, so the support loss and its gradient are exactly 0.
        separable = SimpleNamespace(
            support_set=(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]), None),
            query_set=task.query_set, num_ways=2, num_shots=2, dataset="SYNTHETIC_RESNET")
        fitted = decomposition.evaluate_task(learner, separable, 2, 1.0, check=True)
        self.assertEqual((fitted["support_loss_start"], fitted["zero_gradient_step1"],
                          fitted["norm_ratio_step1"], fitted["cos_g_pg_step1"]),
                         (0.0, 1, None, None))
        self.assertEqual(sum(fitted[f] for f in decomposition.DIVERGED_FIELDS), 0)
        plain = decomposition.adapt_variant(
            learner, task.support_set[0], task.support_set[1], 2, "plain", 1.0)[0]
        self.assertFalse(torch.equal(plain[0], fast[0]))
        self.assertFalse(torch.equal(plain[0], list(learner.learner.parameters())[0]))

    def reference_rows(self, tasks):
        rows = []
        for i, t in enumerate(tasks, 1):
            predictor = self.learner.fit((*t.support_set, t.num_ways, t.num_shots))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "task.predict"
                np.savetxt(path, predictor.predict(t.query_set[0]), fmt="%f")
                scored = accuracy(t.query_set[1].numpy(), read_results_file(str(path)))
            rows.append(dict(task_id=i, dataset=t.dataset, num_ways=t.num_ways,
                             num_shots=t.num_shots, accuracy=scored))
        return rows

    def write_reference(self, path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

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
            self.write_reference(reference, rows)
            out, stdout = root / "decomposition.csv", io.StringIO()
            patches = (patch.object(baseline, "make_encoder", side_effect=tiny_encoder),
                       patch.object(decomposition, "prepare_datasets_information",
                                    return_value=(None, None, {"test": 1})),
                       patch.object(decomposition, "create_datasets", return_value=self.datasets))
            arguments = ["--checkpoint", str(root / "max-va.pth"), "--input_data_dir", "synthetic",
                         "--out_csv", str(out), "--seed", "93", "--test_tasks_per_dataset", "2",
                         "--reference_task_results", str(reference), "--self_check_tasks", "3"]
            with patches[0], patches[1] as prepare, patches[2], redirect_stdout(stdout):
                decomposition.main(arguments)
            prepare.assert_called_once_with("synthetic", self.learner.config["validation_datasets"], 93, False)
            text = stdout.getvalue()
            self.assertIn("saved beta=0.75; gamma=1", text)
            self.assertIn("Self-check PASSED on 3 tasks", text)
            self.assertIn("Reference parity PASSED (full, gamma=1): 4 tasks", text)
            self.assertIn("full-mag_layer", text)
            with out.open() as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(tuple(written[0]), decomposition.FIELDS)
            self.assertEqual(len(written), 4)
            for row, expected, task in zip(written, rows, tasks):
                self.assertEqual(float(row["acc_full"]), expected["accuracy"])
                self.assertEqual((row["dataset"], int(row["num_ways"]), int(row["num_shots"])),
                                 (task.dataset, task.num_ways, task.num_shots))
            # The reference belongs to gamma=1: another gamma must not pass as parity
            # unless every task happens to score the same, and a wrong reference stops.
            rows[0]["accuracy"] = rows[0]["accuracy"] + .05
            self.write_reference(reference, rows)
            failed = root / "failed.csv"
            with patches[0], patches[1], patches[2], redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Reference parity mismatch at task 1"):
                    decomposition.main(arguments[:5] + [str(failed)] + arguments[6:])
            self.assertFalse(failed.exists())
            # Existing outputs and checkpoints are never overwritten.
            for existing in (out, root / "max-va.pth"):
                content = existing.read_bytes()
                with patch.object(decomposition.MyLearner, "load") as load, \
                        patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                    decomposition.main(arguments[:5] + [str(existing)] + arguments[6:])
                load.assert_not_called()
                self.assertEqual(existing.read_bytes(), content)
            for bad in (["--gamma", "nan"], ["--gamma", "-1"], ["--self_check_tasks", "-1"]):
                with patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                    decomposition.main(arguments[:5] + [str(root / "bad.csv")] + arguments[6:] + bad)
            for saved, module in zip(before, (self.learner.learner, self.learner.transport)):
                for key, value in module.state_dict().items():
                    torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
