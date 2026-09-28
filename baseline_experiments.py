"""Notebook entry point; executes only when explicitly called."""
from pathlib import Path
import shutil
import subprocess
import sys


def run_baseline_experiment(baseline="fo_proto_lr_hybrid_ema_warmup", *, input_data_dir,
                            output_dir, data_seed=93, model_seed=98, drive_results_dir=None,
                            **runner_options):
    if model_seed != 98:
        raise ValueError("This controlled comparison uses model seed 98")
    root = Path(__file__).resolve().parent
    submission = root / "baselines" / baseline
    if not (submission / "config.json").is_file():
        raise ValueError("Unknown baseline")
    output = Path(output_dir).resolve()
    options = dict(seed=data_seed, input_data_dir=str(Path(input_data_dir).resolve()),
        submission_dir=str(submission), output_dir_ingestion=str(output/"ingestion"),
        output_dir_scoring=str(output/"scoring"), **runner_options)
    def execute(module, arguments):
        command = [sys.executable, "-m", module] + [
            "--%s=%s" % (key, str(value).lower() if isinstance(value, bool) else value)
            for key, value in arguments.items()]
        subprocess.run(command, cwd=root, check=True)

    if baseline == "fo_proto_lr_hybrid_ema_warmup":
        common_keys = {"seed", "verbose", "debug_mode", "overwrite_previous_results",
                       "test_tasks_per_dataset", "input_data_dir"}
        ingestion_keys = common_keys | {"image_size", "max_time", "save_train_raw_outputs",
                                       "submission_dir", "output_dir_ingestion"}
        scoring_keys = common_keys | {"private_information", "output_dir_scoring"}
        unknown = set(options) - ingestion_keys - scoring_keys
        if unknown:
            raise ValueError("Unsupported runner options: " + ", ".join(sorted(unknown)))
        # Match cdmetadl.run defaults; ingestion's direct default is different.
        options.setdefault("debug_mode", 1)
        execute("baselines.fo_proto_lr_hybrid_ema_warmup.experiment_ingestion",
                {k: v for k, v in options.items() if k in ingestion_keys})
        scoring = {k: v for k, v in options.items() if k in scoring_keys}
        scoring["results_dir"] = str(output / "ingestion")
        execute("cdmetadl.scoring.scoring", scoring)
    else:
        execute("cdmetadl.run", options)
    if drive_results_dir is not None:
        shutil.copytree(output,Path(drive_results_dir)/output.name,dirs_exist_ok=True)
    return output
