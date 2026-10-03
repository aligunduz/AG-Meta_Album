"""Synthetic regressions for legacy metadata and consistent diagnostic gamma."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_gamma_policy import Tiny
import model as baseline
import eval_beta_validation as validation
import eval_gradient_reliability as reliability
import eval_gradient_alignment as alignment
from task_transport import TaskConditionedTransport


class EvaluationGammaRegressions(unittest.TestCase):
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
        self.config["method_config"].update(encoder_lr=.07, classifier_lr=.13, grad_clip=.005)
        encoder = Tiny().double().eval()
        transport = TaskConditionedTransport(encoder, self.config).eval()
        with torch.no_grad():
            transport.u["0"].normal_(0, .7)
            transport.gate_net.out.weight.normal_(0, .5)
            transport.gate_net.out.bias.normal_(0, .3)
        self.learner = baseline.MyLearner()
        self.learner.learner, self.learner.transport = encoder, transport
        self.learner.config, self.learner.dev = self.config, torch.device("cpu")

    def test_old_checkpoint_and_metadata_pass_script_check_but_other_changes_fail(self):
        old_config = copy.deepcopy(self.config)
        old_config["method_config"].pop("train_gamma")
        old_config["method_config"].pop("eval_gamma")
        metadata = dict(baseline="fo_proto_tclrsgmaml", data_seed=93,
                        meta_train_dataset_names=["TRAIN"],
                        meta_validation_dataset_names=["VALID"],
                        meta_test_dataset_names=["TEST"], baseline_config=old_config)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            directory = Path(directory)
            checkpoint = directory / "max-va.pth"
            torch.save(dict(format_version=1, method="fo-proto-tclrsgmaml", config=old_config,
                            model_args=dict(num_classes=2, dev="cpu"),
                            state=baseline.snapshot(self.learner.learner, self.learner.transport),
                            best_validation_accuracy=.5), checkpoint)
            manifest = directory / "run_metadata.json"
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            before = checkpoint.read_bytes(), manifest.read_bytes()

            class ReachedSplitCheck(Exception):
                pass

            # Exercise the actual legacy load and script path, stopping before
            # dataset creation or evaluation once config matching has succeeded.
            with patch.object(baseline, "make_encoder", side_effect=lambda args: Tiny().double()), \
                    patch.object(baseline, "read_config", return_value=self.config), \
                    patch.object(torch.cuda, "is_available", return_value=False), \
                    patch("cdmetadl.helpers.general_helpers.prepare_datasets_information",
                          side_effect=ReachedSplitCheck) as split:
                loaded = baseline.MyLearner()
                loaded.load(checkpoint)
                for gamma in (2, 1):
                    with self.subTest(eval_gamma=gamma):
                        self.config["method_config"]["eval_gamma"] = gamma
                        with self.assertRaises(ReachedSplitCheck):
                            validation.main(["--checkpoint", str(checkpoint), "--run_metadata", str(manifest),
                                             "--input_data_dir", "synthetic", "--seed", "93",
                                             "--tasks_per_dataset", "1", "--out_csv", str(directory / "unused.csv")])
                self.assertEqual(split.call_count, 2)
            self.assertEqual((checkpoint.read_bytes(), manifest.read_bytes()), before)
            self.assertFalse((directory / "unused.csv").exists())

        for section, key in (("lrsg", "beta"), ("lrsg", "rank"),
                ("method_config", "encoder_lr"), ("method_config", "classifier_lr"),
                ("method_config", "inner_steps"), ("method_config", "train_gamma"),
                ("task_conditioning", "scalar_delta_scale"), ("train_config", "k")):
            changed = copy.deepcopy(loaded.config)
            changed[section][key] += 1
            with self.subTest(field=f"{section}.{key}"), self.assertRaises(ValueError):
                validation.check_run_config(changed, old_config)
        for operation in ("added", "removed"):
            changed = copy.deepcopy(loaded.config)
            if operation == "added":
                changed["method_config"]["unexpected"] = 1
            else:
                changed["method_config"].pop("grad_clip")
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                validation.check_run_config(changed, old_config)
        self.assertEqual(metadata["baseline_config"], old_config)
        self.assertNotIn("train_gamma", old_config["method_config"])

    def test_all_reliability_and_alignment_calls_share_configured_eval_gamma(self):
        labels = torch.arange(5).repeat_interleave(2)
        a = torch.randn(10, 3, dtype=torch.float64)
        b = torch.randn(10, 3, dtype=torch.float64)
        query = torch.randn(15, 3, dtype=torch.float64)
        targets = torch.arange(5).repeat_interleave(3)
        sets = {"query": (query, targets), "A": (a, labels), "B": (b, labels)}
        reliability_task = dict(task_id=0, dataset="synthetic", class_ids=list(range(5)),
                                query="query", pairs=[dict(pair_id=0, A="A", B="B")])
        alignment_task = dict(task_id=0, dataset="synthetic", class_ids=list(range(5)),
                              query="query", repeats=[dict(repeat_id=0, support10="A",
                                  shots={"2": dict(positions_in_support10=list(range(10)))})])
        original = self.learner.transport.transport_gradient
        results = {}
        for gamma in (2, 1):
            self.config["method_config"]["eval_gamma"] = gamma
            reference = self.learner.fit((a, labels, None, 5, 2))
            logits = self.learner.learner.forward_weights(query, reference.weights)
            expected_loss = F.cross_entropy(logits, targets).item()
            expected_accuracy = (logits.argmax(1) == targets).double().mean().item()
            seen = []

            def observe(name, gradient, condition, *, gamma):
                seen.append(gamma)
                return original(name, gradient, condition, gamma=gamma)

            load = lambda dataset, selected, classes: sets[selected]
            before = copy.deepcopy(self.config)
            with self.subTest(gamma=gamma), \
                    patch.object(self.learner.transport, "transport_gradient", side_effect=observe), \
                    patch.object(reliability, "load_set", side_effect=load), \
                    patch.object(reliability, "pair_measurements", wraps=reliability.pair_measurements) as paired:
                rows, scores, _ = reliability.evaluate_task(self.learner, None, reliability_task, 1e-20)
                self.assertEqual(paired.call_args.kwargs["gamma"], gamma)
                self.assertEqual(len(seen), len(self.learner.transport.names) * (6 + 4 * reliability.STEPS))
                self.assertTrue(all(value == gamma for value in seen))
                self.assertAlmostEqual(scores[0]["query_ce_lr_on"], expected_loss, places=14)
                self.assertEqual(scores[0]["accuracy_lr_on"], expected_accuracy)
            results[gamma] = dict(fixed_energy=rows[0]["fixed_mean_sq"])
            seen.clear()
            with self.subTest(gamma=gamma), \
                    patch.object(self.learner.transport, "transport_gradient", side_effect=observe), \
                    patch.object(alignment, "load_set", side_effect=load), \
                    patch.object(alignment, "alignment", wraps=alignment.alignment) as measured:
                repeats, _ = alignment.evaluate_task(self.learner, None, alignment_task, (2,), 1e-10)
                self.assertEqual(measured.call_args.kwargs["gamma"], gamma)
                self.assertEqual(len(seen), len(self.learner.transport.names) * (2 + 2 * reliability.STEPS))
                self.assertTrue(all(value == gamma for value in seen))
                self.assertAlmostEqual(repeats[0]["query_ce_lr_on"], expected_loss, places=14)
                self.assertEqual(repeats[0]["accuracy_lr_on"], expected_accuracy)
            results[gamma]["full_norm"] = repeats[0]["norm_full"]
            self.assertEqual(self.config, before)
            self.assertEqual(self.learner.transport.beta, .75)
        # Nonzero U/conditioning ensure a missing gamma changes real measurements.
        self.assertNotEqual(results[1]["fixed_energy"], results[2]["fixed_energy"])
        self.assertNotEqual(results[1]["full_norm"], results[2]["full_norm"])


if __name__ == "__main__":
    unittest.main()
