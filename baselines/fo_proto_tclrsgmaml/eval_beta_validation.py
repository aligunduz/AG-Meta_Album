#!/usr/bin/env python3
r"""Paired gamma=1 vs gamma=2 on the original run's meta-validation domains.

Put this file beside eval_beta_sweep.py in baselines/fo_proto_tclrsgmaml.
Run from the AG-Meta_Album repository root, for example:

    python baselines/fo_proto_tclrsgmaml/eval_beta_validation.py \
        --checkpoint /path/to/seed93/model/max-va.pth \
        --run_metadata /path/to/seed93/run_metadata.json \
        --input_data_dir /content/meta_album_final --seed 93 \
        --tasks_per_dataset 200 --out_csv /content/gamma_validation_seed93.csv

The split seed stays equal to the training run's data_seed. The default episode
seed is data_seed + 10000, so sampling is distinct from the original validation
schedule. Validation DOMAINS are reused; these are tuning data, not a new test
benchmark. Episodes use the final protocol's 2-20 ways, 1-20 shots, 20 query
images/class, and image size 128. Every gamma uses the same task tensors and
restarts adaptation from the loaded checkpoint. Only transport.beta is scaled,
after load; training and checkpoint files are untouched.

The companion JSON records splits, checkpoint/config/code hashes, runtime,
and a paired bootstrap interval conditional on this checkpoint. This interval
does not measure uncertainty from training seeds or unseen domains. Seed 93
is a pilot; repeat with checkpoints/data seeds 97 and 101 before fixing gamma.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


EPISODES = dict(N=None, min_N=2, max_N=20, k=None, min_k=1, max_k=20,
                query_images_per_class=20)
FIELDS = ("task_id", "dataset", "num_ways", "num_shots", "acc_steps0",
          "acc_gamma_1", "acc_gamma_2", "delta_gamma2_minus_gamma1",
          "class_ids")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_run_metadata(path, seed):
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if metadata.get("data_seed") != seed:
        raise ValueError("--seed must equal run_metadata.data_seed; it selects the split")
    if metadata.get("baseline") != "fo_proto_tclrsgmaml":
        raise ValueError("Expected the corresponding fo_proto_tclrsgmaml run_metadata.json")
    fields = ("meta_train_dataset_names", "meta_validation_dataset_names",
              "meta_test_dataset_names")
    groups = []
    for field in fields:
        names = metadata.get(field)
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) for name in names)
                or len(names) != len(set(names))):
            raise ValueError(f"Invalid or missing metadata.{field}")
        groups.append(set(names))
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Training/validation/test domain manifests overlap")
    if not isinstance(metadata.get("baseline_config"), dict):
        raise ValueError("Metadata must include baseline_config")
    return metadata


def check_recreated_split(infos, metadata):
    fields = ("meta_train_dataset_names", "meta_validation_dataset_names",
              "meta_test_dataset_names")
    for info, field in zip(infos, fields):
        if set(info) != set(metadata[field]):
            raise ValueError(f"Recreated {field} differs from the original run")


def summarize(rows):
    import numpy as np
    groups = {"ALL": rows}
    for domain in sorted({row["dataset"] for row in rows}):
        groups[f"domain:{domain}"] = [row for row in rows if row["dataset"] == domain]
    for label, low, high in (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5),
                             ("6-10", 6, 10), ("11-20", 11, 20)):
        groups[f"shot:{label}"] = [row for row in rows if low <= row["num_shots"] <= high]
    result = {}
    for name, selected in groups.items():
        if not selected:
            continue
        result[name] = dict(
            n=len(selected),
            gamma1_accuracy_percent=float(100 * np.mean([r["acc_gamma_1"] for r in selected])),
            gamma2_accuracy_percent=float(100 * np.mean([r["acc_gamma_2"] for r in selected])),
            delta_pp=float(100 * np.mean([r["delta_gamma2_minus_gamma1"] for r in selected])))
    # Paired task resampling within each validation domain preserves its weight.
    rng = np.random.default_rng(20261002)
    boot = np.zeros(4000)
    for domain in sorted({r["dataset"] for r in rows}):
        delta = np.array([r["delta_gamma2_minus_gamma1"] for r in rows if r["dataset"] == domain])
        for start in range(0, len(boot), 100):
            stop = min(start + 100, len(boot))
            indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
            boot[start:stop] += delta[indices].sum(axis=1) / len(rows)
    result["ALL"]["paired_task_bootstrap_ci95_pp"] = (100 * np.quantile(boot, [.025, .975])).tolist()
    result["ALL"]["interval_scope"] = "Evaluation tasks, conditional on this checkpoint and validation domains"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run_metadata", required=True)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--seed", type=int, required=True, help="Original training data/split seed")
    parser.add_argument("--episode_seed", type=int, help="Defaults to seed+10000; does not change the domain split")
    parser.add_argument("--tasks_per_dataset", type=int, default=200)
    args = parser.parse_args(argv)
    episode_seed = args.seed + 10000 if args.episode_seed is None else args.episode_seed
    if not (0 <= args.seed < 2**32 and 0 <= episode_seed < 2**32):
        parser.error("Seeds must be uint32")
    if episode_seed == args.seed or args.tasks_per_dataset < 1:
        parser.error("Use a fresh episode_seed and a positive tasks_per_dataset")
    output = Path(args.out_csv)
    summary_path = output.with_suffix(".summary.json")
    if output.suffix.lower() != ".csv":
        parser.error("--out_csv must have a .csv extension")
    if output.exists() or summary_path.exists():
        parser.error("CSV and companion summary must be new files")
    try:
        metadata = read_run_metadata(args.run_metadata, args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    baseline_dir = Path(__file__).resolve().parent
    if not (baseline_dir / "eval_beta_sweep.py").is_file():
        parser.error("Put this file in baselines/fo_proto_tclrsgmaml beside eval_beta_sweep.py")
    sys.path.insert(0, str(baseline_dir.parents[1]))
    sys.path.insert(0, str(baseline_dir))
    from eval_beta_sweep import (MyLearner, evaluate_task, new_output_csv,
                                validate_checkpoint_config)
    from cdmetadl.helpers.general_helpers import prepare_datasets_information
    from cdmetadl.ingestion.image_dataset import create_datasets
    from cdmetadl.ingestion.data_generator import CompetitionDataLoader
    import torch
    import numpy as np

    started = time.perf_counter()
    learner = MyLearner()
    learner.load(args.checkpoint)
    validate_checkpoint_config(learner.config)
    if learner.config != metadata["baseline_config"]:
        raise ValueError("Loaded checkpoint config differs from the original run metadata")
    infos = prepare_datasets_information(args.input_data_dir,
                                         learner.config["validation_datasets"], args.seed, False)
    check_recreated_split(infos, metadata)
    datasets = create_datasets(infos[1], 128)  # ONLY meta-validation domains
    expected = len(datasets) * args.tasks_per_dataset
    if not expected:
        raise ValueError("No meta-validation datasets")
    # test_generator=True controls balanced iteration per domain, not its split.
    loader = CompetitionDataLoader(datasets, EPISODES, episode_seed, test_generator=True)
    print(f"Validation domains: {list(infos[1])}; {expected} paired tasks; "
          f"split_seed={args.seed}; episode_seed={episode_seed}; gammas=[1,2]", flush=True)
    source_paths = [baseline_dir / name for name in (
        "eval_beta_validation.py", "eval_beta_sweep.py", "eval_inner_steps.py", "model.py",
        "helpers_fo_proto_tclrsgmaml.py", "low_rank_transport.py", "task_transport.py", "network.py")]
    from cdmetadl.ingestion import data_generator, image_dataset
    source_paths.extend([Path(data_generator.__file__), Path(image_dataset.__file__)])
    info = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                original_run=metadata, split="meta-validation", split_seed=args.seed,
                episode_seed=episode_seed, tasks_per_dataset=args.tasks_per_dataset,
                episode_config=EPISODES, image_size=128, gammas=[1, 2],
                checkpoint_path=str(Path(args.checkpoint).resolve()),
                checkpoint_sha256=sha256_file(Path(args.checkpoint) / "max-va.pth"
                                              if Path(args.checkpoint).is_dir() else args.checkpoint),
                saved_beta=learner.transport.beta, checkpoint_config=learner.config,
                source_sha256={str(path): sha256_file(path) for path in source_paths},
                runtime=dict(python=platform.python_version(), pytorch=torch.__version__,
                             cuda=torch.version.cuda, numpy=np.__version__,
                             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with new_output_csv(output) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for task_id, task in enumerate(loader.generator(args.tasks_per_dataset), 1):
            row = evaluate_task(learner, task, task_id, (1.0, 2.0))
            row["delta_gamma2_minus_gamma1"] = row["acc_gamma_2"] - row["acc_gamma_1"]
            row["class_ids"] = json.dumps(task.original_class_idx.tolist())
            writer.writerow(row)
            rows.append(row)
            if task_id % 100 == 0:
                print(f"{task_id}/{expected} tasks; {time.perf_counter() - started:.1f}s", flush=True)
        counts = Counter(row["dataset"] for row in rows)
        if len(rows) != expected or counts != Counter({d: args.tasks_per_dataset for d in infos[1]}):
            raise ValueError("Validation task count/domain mismatch")
        if learner.transport.beta != info["saved_beta"]:
            raise RuntimeError("Temporary beta scale was not restored")
        info.update(summary=summarize(rows), elapsed_seconds=time.perf_counter() - started,
                    evidence_scope="Validation tuning for one checkpoint; compare all three data seeds before fixing gamma")
        with new_output_csv(summary_path) as summary_handle:
            json.dump(info, summary_handle, ensure_ascii=False, indent=2, allow_nan=False)
    total = info["summary"]["ALL"]
    low, high = total["paired_task_bootstrap_ci95_pp"]
    print(f"gamma=1: {total['gamma1_accuracy_percent']:.4f}%; "
          f"gamma=2: {total['gamma2_accuracy_percent']:.4f}%; "
          f"delta: {total['delta_pp']:+.4f} pp; conditional task CI95: [{low:+.4f}, {high:+.4f}]")
    print(f"Wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
