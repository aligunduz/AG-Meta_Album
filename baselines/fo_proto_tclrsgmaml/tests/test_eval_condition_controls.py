"""CPU/synthetic checks; no downloaded data or trained checkpoint is required."""
import copy
import csv
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_condition_controls as controls
import model as baseline
from task_transport import TaskConditionedTransport


class Tiny(nn.Module):
    in_features = 4

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.model = nn.ModuleDict({"out": nn.Identity()})

    def forward_weights(self, x, weights, embedding=False):
        features = F.linear(x, weights[0], weights[1]).tanh()
        return features if embedding else F.linear(features, weights[-2], weights[-1])


class ConditionControlTests(unittest.TestCase):
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
        self.config["task_conditioning"]["scalar_delta_scale"] = 0.0
        self.config["method_config"].update(inner_steps=3, encoder_lr=.07,
                                          classifier_lr=.13, grad_clip=.05)
        encoder = Tiny().double().eval()
        transport = TaskConditionedTransport(encoder, self.config).eval()
        with torch.no_grad():
            transport.u["0"].normal_(0, .7)
            transport.gate_net.out.weight.normal_(0, .5)
            transport.gate_net.out.bias.normal_(0, .3)
        self.learner = baseline.MyLearner()
        self.learner.learner, self.learner.transport = encoder, transport
        self.learner.config = copy.deepcopy(self.config)
        self.learner.state = baseline.snapshot(encoder, transport)
        self.learner.dev = torch.device("cpu")
        self.learner.model_args = dict(num_classes=2, dev="cpu", num_blocks=18, pretrained=False)
        self.learner.best_score = .5
        self.codec = controls.RankCodec(self.learner.state["architecture"])
        self.tasks = []
        for i, dataset in enumerate(("DOG", "DOG", "DOG", "TEX", "TEX", "TEX")):
            ways, shots = 2 + i % 2, 1 + i % 2
            labels = torch.arange(ways).repeat(shots)
            query_labels = torch.arange(ways).repeat(2)
            self.tasks.append(SimpleNamespace(
                dataset=dataset, num_ways=ways, num_shots=shots,
                support_set=(torch.randn(len(labels), 3, dtype=torch.float64) + i,
                             labels, labels + 10 * i),
                query_set=(torch.randn(len(query_labels), 3, dtype=torch.float64),
                           query_labels, query_labels + 10 * i),
                original_class_idx=np.arange(ways) + 10 * i))

    def test_exact_derangements_determinism_multisets_and_prefix(self):
        labels = ["DOG"] * 5 + ["TEX"] * 4 + ["BIRD"] * 3
        coefficients = np.arange(12 * 7).reshape(12, 7)
        # Include repeated vectors; preserve multiplicity, not just the set.
        coefficients[2] = coefficients[1]
        np_state = np.random.get_state()
        globals_, withins = controls.make_permutations(labels, 20, 12345, coefficients)
        again_g, again_w = controls.make_permutations(labels, 99, 12345, coefficients)
        np.testing.assert_array_equal(globals_, again_g[:20])
        np.testing.assert_array_equal(withins, again_w[:20])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(len({tuple(row) for row in globals_}), 20)
        self.assertEqual(len({tuple(row) for row in withins}), 20)
        for permutation in globals_:
            controls.check_permutation(permutation, labels, coefficients)
        for permutation in withins:
            controls.check_permutation(permutation, labels, coefficients, within=True)
        other, _ = controls.make_permutations(labels, 20, 999)
        self.assertFalse(np.array_equal(globals_, other))
        np.testing.assert_array_equal(controls.derangement(2, np.random.default_rng(0)), [1, 0])
        for labels in (["DOG"], ["DOG", "DOG", "TEX"]):
            with self.assertRaisesRegex(ValueError, "singleton"):
                controls.make_permutations(labels, 2, 1)
        with self.assertRaisesRegex(ValueError, "self-assignment"):
            controls.check_permutation(np.arange(4), ["D"] * 4)
        with self.assertRaisesRegex(ValueError, "crosses datasets"):
            controls.check_permutation(np.array([2, 3, 0, 1]), ["D", "D", "T", "T"], within=True)
        with self.assertRaisesRegex(ValueError, "Too few distinct"):
            controls.make_permutations(["DOG", "DOG", "TEX", "TEX"], 2, 1)
        self.assertEqual([controls.derangement_count(n, 1000) for n in range(7)],
                         [1, 0, 1, 2, 9, 44, 265])

    def test_layout_round_trip_multiple_layers_numeric_keys_and_variable_ranks(self):
        encoder = nn.Sequential(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6),
                                nn.Linear(6, 5), nn.Linear(5, 3), nn.Linear(3, 2),
                                nn.Linear(2, 1)).double()
        encoder.in_features = 1
        transport = TaskConditionedTransport(encoder, self.config).eval()
        codec = controls.RankCodec(transport.architecture())
        self.assertEqual([row["key"] for row in codec.rows], ["0", "4", "6", "8", "10"])
        self.assertEqual([row["rank"] for row in codec.rows], [4, 4, 3, 2, 1])
        vector = torch.arange(codec.size, dtype=torch.float64) * -.3
        mapped = codec.reconstruct(vector, next(encoder.parameters()))
        torch.testing.assert_close(codec.flatten(mapped), vector, rtol=0, atol=0)
        codec.check_round_trip(vector, mapped)
        # Absolute GateNet offsets must not be used directly on the residual-only vector.
        torch.testing.assert_close(mapped["10"], vector[-1:], rtol=0, atol=0)
        bad = copy.deepcopy(transport.architecture())
        bad["task_conditioning"]["rank_layout"][1]["start"] += 1
        with self.assertRaisesRegex(ValueError, "contiguous"):
            controls.RankCodec(bad)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            codec.reconstruct(vector[:-1], next(encoder.parameters()))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            codec.reconstruct(vector * float("nan"), next(encoder.parameters()))

    def test_own_exact_normal_equivalence_and_diagnostics_are_observational(self):
        guard = controls.StateGuard(self.learner)
        for task in self.tasks:
            result = controls.evaluate_task(self.learner, task, self.codec)
            expected = self.learner.fit((*task.support_set, task.num_ways, task.num_shots)).predict(task.query_set[0])
            np.testing.assert_array_equal(result[3], expected)
            without = controls.evaluate_task(self.learner, task, self.codec, diagnostics=False)
            np.testing.assert_array_equal(result[3], without[3])
            torch.testing.assert_close(result[1], without[1], rtol=0, atol=0)
            self.assertGreater(result[2]["correction_to_gradient_ratio"], 0)
            self.assertEqual(result[2]["inner_steps"], 3)
            guard.check()

    def test_mean_and_shuffle_override_bypass_gate_net_and_keep_scalar_zero(self):
        vectors = torch.stack([controls.evaluate_task(self.learner, task, self.codec)[1]
                               for task in self.tasks])
        self.assertFalse(torch.equal(vectors[0], vectors[1]))
        mean = vectors.mean(0)
        permutation = [3, 4, 5, 0, 1, 2]
        transport = self.learner.transport
        original = transport.transport_gradient
        seen = []

        def observe(name, gradient, conditioning):
            self.assertEqual(int(conditioning[0].count_nonzero()), 0)
            seen.append(self.codec.flatten(conditioning[1]))
            return original(name, gradient, conditioning)

        with patch.object(transport.gate_net, "forward", side_effect=AssertionError("GateNet override leak")), \
                patch.object(transport, "transport_gradient", side_effect=observe):
            for i, task in enumerate(self.tasks):
                for expected in (mean, vectors[permutation[i]]):
                    seen.clear()
                    result = controls.evaluate_task(self.learner, task, self.codec, expected)
                    torch.testing.assert_close(result[1], expected, rtol=0, atol=0)
                    self.assertTrue(all(torch.equal(v, expected) for v in seen))
        # Supplying its own stored vector reproduces OWN exactly as well.
        own = controls.evaluate_task(self.learner, self.tasks[0], self.codec)
        reapplied = controls.evaluate_task(self.learner, self.tasks[0], self.codec, vectors[0])
        np.testing.assert_array_equal(own[3], reapplied[3])

    def test_support_only_single_conditioning_call_query_independence(self):
        task = self.tasks[0]
        seen = []
        hook = self.learner.transport.gate_net.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].clone()))
        try:
            own = controls.evaluate_task(self.learner, task, self.codec)
            changed = copy.deepcopy(task)
            changed.query_set = (changed.query_set[0] + 100, changed.query_set[1].flip(0), changed.query_set[2])
            other = controls.evaluate_task(self.learner, changed, self.codec)
        finally:
            hook.remove()
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(not value.requires_grad for value in seen))
        expected = self.learner.learner.forward_weights(
            task.support_set[0], list(self.learner.learner.parameters()), embedding=True).mean(0).detach()
        torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(own[1], other[1], rtol=0, atol=0)

    def test_first_order_clipping_lrs_head_updates_no_optimizer_or_parameter_mutation(self):
        transport, model = self.learner.transport, self.learner.learner
        config = self.config["method_config"]
        original_grad = torch.autograd.grad
        original_transport = transport.transport_gradient
        previous, transported = [], []
        guard = controls.StateGuard(self.learner)

        def observe_grad(loss, fast, **kwargs):
            self.assertIs(kwargs["create_graph"], False)
            self.assertIs(kwargs["retain_graph"], True)
            if previous:
                old, clipped = previous[-1]
                for current, before, grad in zip(fast[-2:], old[-2:], clipped[-2:]):
                    torch.testing.assert_close(current, before - config["classifier_lr"] * grad, rtol=0, atol=0)
                for current, before, grad in zip(fast[:-2], old[:-2], transported[-len(transport.names):]):
                    torch.testing.assert_close(current, before - config["encoder_lr"] * grad, rtol=0, atol=0)
            grads = original_grad(loss, fast, **kwargs)
            self.assertTrue(all(grad.grad_fn is None for grad in grads))
            previous.append(([w.detach().clone() for w in fast],
                             [g.clamp(-config["grad_clip"], config["grad_clip"]) for g in grads]))
            return grads

        def observe_transport(name, gradient, conditioning):
            self.assertIn(name, dict(model.named_parameters()))
            self.assertLessEqual(float(gradient.abs().max()), config["grad_clip"])
            value = original_transport(name, gradient, conditioning)
            transported.append(value.detach().clone())
            return value

        with patch.object(torch.autograd, "grad", side_effect=observe_grad), \
                patch.object(transport, "transport_gradient", side_effect=observe_transport), \
                patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("optimizer construction")), \
                patch.object(torch.optim.Adam, "step", side_effect=AssertionError("optimizer step")):
            controls.evaluate_task(self.learner, self.tasks[0], self.codec)
        self.assertEqual(len(previous), config["inner_steps"])
        self.assertEqual(len(transported), len(transport.names) * config["inner_steps"])
        guard.check()
        with torch.no_grad():
            transport.u["0"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "State changed"):
            guard.check()

    def test_episode_replay_checks_full_content_labels_metadata_and_count(self):
        replay = controls.CheckedEpisodes(lambda: iter(self.tasks), 94, len(self.tasks))
        list(replay.iterate(record=True))
        list(replay.iterate())
        for field in ("dataset", "num_ways", "num_shots", "support_set", "query_set", "original_class_idx"):
            changed = copy.deepcopy(self.tasks)
            if field in ("support_set", "query_set"):
                values = getattr(changed[0], field)
                # Pixels differ even though dataset/ways/shots/labels are identical.
                setattr(changed[0], field, (values[0] + 1, values[1], values[2]))
            elif field == "dataset":
                changed[0].dataset = "OTHER"
            else:
                setattr(changed[0], field, getattr(changed[0], field) + 1)
            replay.factory = lambda: iter(changed)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                list(replay.iterate())
        for field in ("support_set", "query_set"):
            changed = copy.deepcopy(self.tasks)
            x, y, original = getattr(changed[0], field)
            setattr(changed[0], field, (x, y.flip(0), original))
            replay.factory = lambda: iter(changed)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                list(replay.iterate())
        replay.factory = lambda: iter(self.tasks[:-1])
        with self.assertRaisesRegex(ValueError, "number of test episodes"):
            list(replay.iterate())
        replay.factory = lambda: iter(self.tasks + [self.tasks[0]])
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            list(replay.iterate())

    def test_replay_rng_is_deterministic_and_restores_caller(self):
        before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
        with controls.replay_rng(94):
            expected = (random.random(), np.random.random(), torch.rand(4))
        with controls.replay_rng(94):
            self.assertEqual(expected[0], random.random())
            self.assertEqual(expected[1], np.random.random())
            torch.testing.assert_close(expected[2], torch.rand(4), rtol=0, atol=0)
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        torch.testing.assert_close(before[2], torch.random.get_rng_state(), rtol=0, atol=0)

    def test_paired_and_permutation_statistics_ties_and_resolution(self):
        own = np.array([.8, .6, .4, .2])
        mean = np.array([.7, .6, .5, .1])
        stats = controls.paired_statistics(own, mean)
        d = own - mean
        self.assertAlmostEqual(stats["mean_paired_difference"], d.mean())
        self.assertAlmostEqual(stats["standard_error"], d.std(ddof=1) / 2)
        self.assertEqual(stats["fraction_own_greater"], .5)
        self.assertEqual(stats["fraction_own_less"], .25)
        self.assertEqual(stats["fraction_equal"], .25)
        self.assertLess(stats["ci95"][0], d.mean())
        self.assertGreater(stats["ci95"][1], d.mean())
        shuffled = np.stack([own - .1, own, own + .1])
        null = controls.shuffle_statistics(own, shuffled)
        self.assertEqual(null["permutation_p_value"], 3 / 4)
        self.assertEqual(len(null["paired_by_permutation"]), 3)
        minimum = controls.shuffle_statistics(own, np.tile(own - .1, (20, 1)))
        self.assertEqual(minimum["permutation_p_value"], 1 / 21)
        equal = controls.paired_statistics(own, own)
        self.assertEqual(equal["ci95"], [0, 0])
        self.assertIsNone(controls.paired_statistics([.5], [.2])["standard_error"])

    def test_diagnostics_formula_and_zero_correction(self):
        transport = self.learner.transport
        diag = controls.CorrectionDiagnostics(transport)
        gradients, transformed, corrections = [], [], []
        conditioning = transport.condition(torch.ones(4, dtype=torch.float64))
        for name, parameter in self.learner.learner.named_parameters():
            g = torch.randn_like(parameter)
            t = transport.transport_gradient(name, g, conditioning)
            c = t - transport.logits[transport.indices[name]].sigmoid() * g
            gradients.append(g.reshape(-1))
            transformed.append(t.detach().reshape(-1))
            corrections.append(c.detach().reshape(-1))
            diag.record(name, g, t)
        g, t, c = torch.cat(gradients), torch.cat(transformed), torch.cat(corrections)
        values = diag.result()
        self.assertAlmostEqual(values[controls.DIAGNOSTICS[0]], float(c.norm() / g.norm()))
        self.assertAlmostEqual(values[controls.DIAGNOSTICS[1]], float(t.norm() / g.norm()))
        self.assertAlmostEqual(values[controls.DIAGNOSTICS[2]], float(F.cosine_similarity(g, c, dim=0)))
        self.assertAlmostEqual(values[controls.DIAGNOSTICS[3]], float(F.cosine_similarity(g, t, dim=0)))
        with torch.no_grad():
            transport.u["0"].zero_()
        values = controls.evaluate_task(self.learner, self.tasks[0], self.codec)[2]
        self.assertEqual(values["correction_to_gradient_ratio"], 0)
        self.assertIsNone(values["cos_gradient_correction"])

    def test_rejects_non_lr_only_checkpoints_without_rewriting_config(self):
        controls.validate_lr_only(self.learner)
        original = copy.deepcopy(self.learner.config)
        for field, value in (("scalar_delta_scale", 1.), ("low_rank_delta_scale", .5), ("enabled", False)):
            self.learner.config = copy.deepcopy(original)
            self.learner.config["task_conditioning"][field] = value
            with self.assertRaisesRegex(ValueError, "checkpoint|LR-only"):
                controls.validate_lr_only(self.learner)
            self.assertEqual(self.learner.config["task_conditioning"][field], value)

    def test_complete_synthetic_analysis_outputs_and_reproducibility(self):
        guard = controls.StateGuard(self.learner)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            summaries = []
            for repeat in ("first", "second"):
                episodes = controls.CheckedEpisodes(lambda: iter(self.tasks), 94, 6)
                with redirect_stdout(io.StringIO()), \
                        patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("optimizer")):
                    summaries.append(controls.run_analysis(self.learner, episodes, out / repeat, 3, 12345))
            self.assertEqual(summaries[0], summaries[1])
            first = out / "first"
            for name in ("condition_control_results.csv", "condition_shuffle_results.csv",
                         "condition_dataset_results.csv", "condition_control_summary.json",
                         "condition_task_manifest.json"):
                self.assertEqual((first / name).read_bytes(), (out / "second" / name).read_bytes())
            result = json.loads((first / "condition_control_summary.json").read_text())
            self.assertEqual(result["sanity_checks"]["verified_episode_passes"], 8)
            with (first / "condition_control_results.csv").open(newline="") as handle:
                main_rows = list(csv.DictReader(handle))
            with (first / "condition_shuffle_results.csv").open(newline="") as handle:
                shuffle_rows = list(csv.DictReader(handle))
            self.assertEqual(len(main_rows), 6)
            self.assertEqual(len(shuffle_rows), 36)
            with np.load(first / "condition_vectors.npz", allow_pickle=False) as vectors:
                self.assertEqual(vectors["own_delta_c"].shape, (6, self.codec.size))
                # NumPy and torch may sum in different orders; actual MEAN
                # overrides are separately required to match the stored vector exactly.
                np.testing.assert_allclose(vectors["mean_delta_c"], vectors["own_delta_c"].mean(axis=0),
                                           rtol=1e-14, atol=1e-14)
                self.assertEqual(json.loads(str(vectors["rank_layout_json"])), self.codec.rows)
                for row in shuffle_rows:
                    index, source, pid = int(row["task_index"]), int(row["assigned_from_task_index"]), int(row["permutation_id"])
                    key = "global_permutation_indices" if row["mode"] == "shuffle_global" else "within_permutation_indices"
                    self.assertEqual(source, vectors[key][pid, index])
                    self.assertNotEqual(source, index)
                    self.assertAlmostEqual(float(row["difference_from_own"]), float(row["accuracy"]) - float(row["own_accuracy"]))
        guard.check()

    def test_cli_uses_checkpoint_config_test_protocol_and_requested_seed(self):
        # Exercise checkpoint save/load and main's wiring with only synthetic episodes.
        with tempfile.TemporaryDirectory() as directory:
            self.learner.save(directory)
            tasks = copy.deepcopy(self.tasks)
            for task in tasks:
                task.support_set = (task.support_set[0].float(), *task.support_set[1:])
                task.query_set = (task.query_set[0].float(), *task.query_set[1:])
            datasets = [object(), object()]
            loader = SimpleNamespace(generator=lambda count: iter(tasks))
            with patch.object(baseline, "ResNet", Tiny), \
                    patch.object(torch.cuda, "is_available", return_value=False), \
                    patch("cdmetadl.helpers.general_helpers.prepare_datasets_information", return_value=(None, None, {"D": 1, "T": 2})) as info, \
                    patch("cdmetadl.ingestion.image_dataset.create_datasets", return_value=datasets), \
                    patch("cdmetadl.ingestion.data_generator.CompetitionDataLoader", return_value=loader) as factory, \
                    redirect_stdout(io.StringIO()):
                controls.main(["--checkpoint", str(Path(directory) / "max-va.pth"),
                               "--input_data_dir", directory, "--output_dir", str(Path(directory) / "out"),
                               "--test_tasks_per_dataset", "3", "--num_permutations", "1"])
            self.assertEqual(info.call_args.args[2], 94)
            self.assertEqual(factory.call_count, 4)
            for call in factory.call_args_list:
                self.assertEqual(call.args, (datasets, controls.TEST_EPISODES, 94))
                self.assertIs(call.kwargs["test_generator"], True)
            summary = json.loads((Path(directory) / "out/condition_control_summary.json").read_text())
            self.assertEqual(summary["metadata"]["method_config"], self.config["method_config"])
            self.assertEqual(len(summary["metadata"]["checkpoint_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
