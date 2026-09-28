"""Runner routing checks; subprocesses are mocked, no experiment runs."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from baseline_experiments import run_baseline_experiment


class RunnerTests(unittest.TestCase):
    def test_local_protocol_matches_shared_except_two_hooks(self):
        shared = ast.parse((ROOT / "cdmetadl/ingestion/ingestion.py").read_text())
        source = (ROOT / "baselines/fo_proto_lr_hybrid_ema_warmup/experiment_ingestion.py").read_text()
        for hook in (
            '    if hasattr(meta_learner, "set_run_context"):\n        meta_learner.set_run_context(data_seed=SEED)\n',
            '        if hasattr(learner, "record_test_episode"):\n            learner.record_test_episode(task, y_pred, i + 1)\n',
        ):
            self.assertEqual(source.count(hook), 1)
            source = source.replace(hook, "")
        local = ast.parse(source)
        # The local runner has a different module description, same protocol.
        self.assertEqual(ast.dump(ast.Module(body=shared.body[1:], type_ignores=[])),
                         ast.dump(ast.Module(body=local.body[1:], type_ignores=[])))

    def test_hybrid_routes_seed_and_outputs_then_copies(self):
        with tempfile.TemporaryDirectory() as d, patch("baseline_experiments.subprocess.run") as run, patch("baseline_experiments.shutil.copytree") as copy:
            output = run_baseline_experiment(input_data_dir=d, output_dir=Path(d)/"output", data_seed=97,
                drive_results_dir=Path(d)/"drive", private_information=True, image_size=64)
            first, second = [call.args[0] for call in run.call_args_list]
            self.assertEqual(first[2], "baselines.fo_proto_lr_hybrid_ema_warmup.experiment_ingestion")
            self.assertEqual(second[2], "cdmetadl.scoring.scoring")
            for command in (first, second):
                self.assertIn("--seed=97", command)
                self.assertIn("--debug_mode=1", command)
            self.assertIn("--image_size=64", first)
            self.assertNotIn("--private_information=true", first)
            self.assertIn("--private_information=true", second)
            self.assertIn("--results_dir="+str(output/"ingestion"), second)
            copy.assert_called_once()

    def test_existing_baseline_keeps_original_runner(self):
        with patch("baseline_experiments.subprocess.run") as run:
            run_baseline_experiment("fo_proto_tclrsgmaml", input_data_dir="data", output_dir="results")
            self.assertEqual(run.call_args.args[0][2], "cdmetadl.run")
            self.assertEqual(run.call_count, 1)

    def test_ingestion_failure_stops_scoring_and_copy(self):
        with patch("baseline_experiments.subprocess.run", side_effect=subprocess.CalledProcessError(1,"runner")) as run, patch("baseline_experiments.shutil.copytree") as copy:
            with self.assertRaises(subprocess.CalledProcessError):
                run_baseline_experiment(input_data_dir="data", output_dir="results", drive_results_dir="drive")
            self.assertEqual(run.call_count, 1)
            copy.assert_not_called()
