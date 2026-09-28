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


class ColabCompatibilityTests(unittest.TestCase):
    def options(self, baseline, seed):
        from types import SimpleNamespace
        return SimpleNamespace(seed=seed, verbose=True, debug_mode=1, image_size=128,
            max_time=1000, overwrite_previous_results=False, test_tasks_per_dataset=100,
            private_information=False, save_train_raw_outputs=False,
            input_data_dir="/content/data", output_dir_ingestion="/content/run/ingestion",
            output_dir_scoring="/content/run/scoring",
            submission_dir="/content/AG-Meta_Album/baselines/"+baseline)

    def test_unchanged_colab_cli_selects_hybrid_for_any_data_seed(self):
        import cdmetadl.run as runner
        for seed in (93, 95):
            with patch.object(runner,"FLAGS",self.options("fo_proto_lr_hybrid_ema_warmup",seed)), patch.object(runner,"call",return_value=0) as call:
                runner.main([])
                commands=[c.args[0] for c in call.call_args_list]
                self.assertEqual(commands[0][2],"baselines.fo_proto_lr_hybrid_ema_warmup.experiment_ingestion")
                self.assertEqual(commands[1][2],"cdmetadl.scoring.scoring")
                self.assertTrue(all("--seed="+str(seed) in c for c in commands))

    def test_existing_colab_baseline_keeps_shared_ingestion(self):
        import cdmetadl.run as runner
        with patch.object(runner,"FLAGS",self.options("fo_proto_tclrsgmaml",95)), patch.object(runner,"call",return_value=0) as call:
            runner.main([])
            self.assertEqual(call.call_args_list[0].args[0][2],"cdmetadl.ingestion.ingestion")

    def test_hybrid_failure_reaches_notebook(self):
        import cdmetadl.run as runner
        with patch.object(runner,"FLAGS",self.options("fo_proto_lr_hybrid_ema_warmup",93)), patch.object(runner,"call",return_value=2) as call:
            with self.assertRaises(SystemExit) as error:
                runner.main([])
            self.assertEqual(error.exception.code,2)
            self.assertEqual(call.call_count,1)
