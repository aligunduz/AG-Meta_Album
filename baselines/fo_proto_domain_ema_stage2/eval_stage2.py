"""Paired A/B/C/D and domain-by-specialist evaluation on one episode stream."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.fo_proto_domain_ema_stage2.data import (
    loaders, episode_record, file_hash, json_hash, DOMAIN_INDEX, context_for_task, canonical)
from baselines.fo_proto_domain_ema_stage2.experiment import (
    Experiment, compare_logits, source_logits)
from baselines.fo_proto_domain_ema_stage2.model import Stage2Learner
from baselines.fo_proto_domain_ema_stage2.source import reference
from baselines.fo_proto_domain_ema_stage2.reporting import write_summary, error_record


def score(logits, task):
    labels = task.query_set[1]
    correct = int((logits.argmax(1) == labels).sum())
    return correct, labels.numel()


def aggregate(rows):
    result = {}
    for condition in ("A", "B", "C", "D"):
        selected = [row for row in rows if row["condition"] == condition]
        if not selected:
            continue
        ids = sorted({row["dataset"] for row in selected})
        per_dataset = []
        for identifier in ids:
            subset = [row for row in selected if row["dataset"] == identifier]
            per_dataset.append(float(np.mean([row["accuracy"] for row in subset])))
        result[condition] = dict(tasks=len(selected),
                                accuracy=float(np.mean([row["accuracy"] for row in selected])),
                                aggregation="task_mean",
                                query_micro_accuracy=sum(row["correct"] for row in selected) / sum(row["total"] for row in selected),
                                dataset_macro_accuracy=float(np.mean(per_dataset)))
    return result


def paired_bootstrap(rows, seed, samples):
    """Stratified paired task bootstrap; keep all seven datasets fixed.

    Each replicate resamples paired episodes within each dataset, computes
    task accuracy differences B-C, with equal weight for every task.
    This CI is conditional on these seven datasets, not on unseen datasets.
    """
    rng = np.random.default_rng(seed)
    grouped = {}
    for row in rows:
        if row["condition"] in ("B", "C") and row["seen"]:
            pair = grouped.setdefault(row["dataset"], {}).setdefault(row["task_id"], {})
            if row["condition"] in pair:
                raise ValueError("Duplicate task/condition in paired bootstrap")
            pair[row["condition"]] = row
    distribution = np.zeros(samples)
    point_sum = 0.
    task_count = 0
    for episodes in grouped.values():
        pairs = list(episodes.values())
        if any(set(pair) != {"B", "C"} for pair in pairs):
            raise ValueError("Unpaired B/C episodes")
        if any(p["B"]["episode_hash"] != p["C"]["episode_hash"] or
               p["B"]["total"] != p["C"]["total"] for p in pairs):
            raise ValueError("B/C episode identity or query count mismatch")
        differences = np.array([p["B"]["accuracy"] - p["C"]["accuracy"] for p in pairs])
        point_sum += float(differences.sum())
        task_count += len(pairs)
        for i in range(samples):
            indices = rng.integers(0, len(pairs), size=len(pairs))
            distribution[i] += differences[indices].sum()
    if len(grouped) != 7:
        raise ValueError("Bootstrap requires all seven seen datasets")
    distribution /= task_count
    return dict(estimate=point_sum / task_count, ci95=np.quantile(distribution, [.025, .975]).tolist(),
                samples=samples, seed=seed, unit="paired task within fixed dataset strata",
                aggregation="task_mean B-C; fixed dataset strata and original task counts")


def read_old_results(path, expected_tasks):
    """Validate CSV structure before any expensive evaluation."""
    if not path:
        return dict(status="not provided")
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        old = list(csv.DictReader(handle))
    digest = file_hash(path)
    if not old or "task_id" not in old[0]:
        return dict(status="not verifiable", reason="CSV lacks task_id rows", file_sha256=digest)
    old = [row for row in old if row.get("condition", "A") == "A"]
    if not old:
        raise ValueError("Historical CSV has no baseline rows")
    numeric = all(row["task_id"].strip().isdecimal() for row in old)
    identifiers = [int(row["task_id"]) if numeric else row["task_id"].strip() for row in old]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Historical CSV has duplicate task IDs")
    if numeric:
        if set(identifiers) != set(range(1, expected_tasks + 1)):
            raise ValueError(f"Historical task IDs must cover 1..{expected_tasks} exactly")
        if not {"dataset", "num_ways", "num_shots"}.issubset(old[0]):
            raise ValueError("Numeric task IDs require dataset, num_ways and num_shots columns")
    elif not all(len(value) == 64 and all(c in "0123456789abcdef" for c in value) for value in identifiers):
        raise ValueError("Historical task_id must be a native integer or a full episode SHA256")
    for row, identifier in zip(old, identifiers):
        row["task_id"] = identifier
        row["accuracy"] = float(row.get("accuracy", row.get("Accuracy", "nan")))
        if not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1:
            raise ValueError(f"Invalid historical accuracy for task {identifier}")
        if numeric:
            row["dataset"] = canonical(row["dataset"])
            row["num_ways"], row["num_shots"] = int(row["num_ways"]), int(row["num_shots"])
        if row.get("predictions"):
            row["predictions"] = json.loads(row["predictions"])
    return dict(status="ready", mode="task_index" if numeric else "episode_hash",
                file_sha256=digest, rows=old)


def compare_old_results(historical, rows, predictions, atol):
    if historical["status"] != "ready":
        return historical
    mode = historical["mode"]
    key = "task_id" if mode == "task_index" else "episode_hash"
    baseline = [row for row in rows if row["condition"] == "A"]
    expected = {row[key]: row for row in baseline}
    if len(expected) != len(baseline):
        raise ValueError("Repeated episode hashes cannot identify unique historical task occurrences")
    mismatches, compared, prediction_checks, hash_checks = [], 0, 0, 0
    old = historical["rows"]
    if len(old) != len(expected):
        mismatches.append(dict(task_id=None, checks=["historical/current task counts differ"]))
    for row in old:
        identifier = row["task_id"]
        actual = expected.get(identifier)
        failures = []
        if actual is None:
            mismatches.append(dict(task_id=identifier, checks=["missing current episode"]))
            continue
        if mode == "task_index":
            for field in ("dataset", "num_ways", "num_shots"):
                if row[field] != actual[field]:
                    failures.append(f"{field}: historical={row[field]!r}, current={actual[field]!r}")
        if abs(row["accuracy"] - actual["accuracy"]) > atol:
            failures.append(f"accuracy: historical={row['accuracy']}, current={actual['accuracy']}")
        if mode == "episode_hash" or row.get("episode_hash"):
            hash_checks += 1
            if mode == "task_index" and row["episode_hash"] != actual["episode_hash"]:
                failures.append("episode_hash")
        if row.get("predictions"):
            prediction_checks += 1
            if row["predictions"] != predictions[actual["task_id"]]:
                failures.append("predictions")
        if failures:
            mismatches.append(dict(task_id=identifier, checks=failures))
        compared += 1
    return dict(status="failed" if mismatches else "matched", tasks=compared,
                complete=compared == len(expected), mode=mode, mismatches=mismatches,
                prediction_checks=prediction_checks, episode_hash_checks=hash_checks,
                full_episode_identity_verified=not mismatches and hash_checks == len(expected),
                scope=("task index, dataset, ways, shots and accuracy; image identity is not proven by these fields"
                       if mode == "task_index" else "episode hash and accuracy"),
                file_sha256=historical["file_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_checkpoint", required=True, help="Original max-va.pth for its own loader")
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_tasks_per_dataset", type=int, default=100)
    parser.add_argument("--test_seed", type=int, help="Defaults to the recorded stage-1 evaluation seed")
    parser.add_argument("--stage1_results", help="Optional old task CSV")
    args = parser.parse_args()
    if args.test_tasks_per_dataset < 1:
        raise ValueError("test_tasks_per_dataset must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(kind="stage2_evaluation", status="running", phase="setup", completed_tasks=0,
                   checkpoint=args.checkpoint, source_checkpoint=args.source_checkpoint)
    write_summary(output / "summary.json", summary)
    try:
        evaluate(args, output, summary)
        summary["status"] = "passed"
    except BaseException as error:
        summary.update(status="failed", error=error_record(error))
        raise
    finally:
        # Includes partial progress and failures, even if historical comparison
        # or a model/manifest check raises. Existing task CSVs remain available.
        write_summary(output / "summary.json", summary)


def evaluate(args, output, summary):
    experiment = Experiment.load(args.checkpoint)
    if file_hash(args.source_checkpoint) != experiment.provenance["source_sha256"]:
        raise ValueError("Reference checkpoint hash mismatch")
    options = dict(experiment.options)
    if args.test_seed is not None:
        options["test_seed"] = args.test_seed
    (_, _, test_loader), manifest = loaders(args.input_data_dir, experiment.source["config"], options,
                                            experiment.provenance["image_size"])
    if manifest != experiment.manifest:
        raise ValueError("Evaluation split, dataset order, or metadata changed")
    summary["phase"] = "historical_csv_preflight"
    try:
        historical = read_old_results(args.stage1_results, len(manifest["test"]) * args.test_tasks_per_dataset)
    except Exception as error:
        summary["historical_results"] = dict(status="failed", error=error_record(error))
        raise
    summary["historical_results"] = {key: value for key, value in historical.items() if key != "rows"}
    oracle = reference().MyLearner()
    oracle.load(args.source_checkpoint)
    adapter = Stage2Learner()
    adapter.load(args.checkpoint)
    frozen_before = {key: branch.state() for key, branch in experiment.branches.items()}
    frozen_before["all_stepmatched"] = experiment.early.state()
    rows, matrix, identities, baseline_predictions = [], [], [], {}
    comparisons = dict(source_max_logit_error=0., adapter_max_probability_error=0.,
                       source_tasks=0, adapter_tasks=0)
    operations = dict(conditions=0, off_domain_specialists=0, source_checks=0, adapter_checks=0)
    summary.update(phase="episodes", comparisons=comparisons, evaluation_adaptation_operations=operations)
    start = time.perf_counter()
    fields = ["task_id", "episode_hash", "dataset", "num_ways", "num_shots", "domain", "seen", "condition", "branch", "expert_id",
              "correct", "total", "accuracy", "loss", "seconds", "predictions"]
    with (output / "test-manifest.jsonl").open("x", encoding="utf-8") as manifests, \
            (output / "tasks.csv").open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        # Every condition receives the very same materialized task tensors.
        for task_index, task in enumerate(test_loader.generator(args.test_tasks_per_dataset), 1):
            record = episode_record(task)
            record["episode_hash"] = record.pop("task_id")
            record["task_id"] = task_index
            identities.append(dict(task_id=task_index, episode_hash=record["episode_hash"]))
            manifests.write(json.dumps(record) + "\n")
            context = context_for_task(task)
            domain = context["domain"]
            seen = domain in experiment.seen
            outputs = {}
            for condition in ("A", "B", "C", "D"):
                key = experiment.route(domain, condition)
                task_start = time.perf_counter()
                logits, loss = experiment.evaluate(task, key)
                operations["conditions"] += 1
                outputs[condition] = logits
                correct, total = score(logits, task)
                row = dict(task_id=task_index, episode_hash=record["episode_hash"],
                           dataset=record["dataset_id"], domain=domain,
                           num_ways=task.num_ways, num_shots=task.num_shots,
                           seen=seen, condition=condition, branch=key,
                           expert_id=DOMAIN_INDEX[domain] if key.startswith("specialist_") else key,
                           correct=correct, total=total, accuracy=correct / total, loss=loss,
                           seconds=time.perf_counter() - task_start,
                           predictions=json.dumps(logits.argmax(1).tolist()))
                writer.writerow(row)
                rows.append(row)
            baseline_predictions[record["task_id"]] = outputs["A"].argmax(1).tolist()
            expected = source_logits(oracle, task)
            operations["source_checks"] += 1
            error = compare_logits(outputs["A"], expected, options, record["task_id"])
            comparisons["source_max_logit_error"] = max(comparisons["source_max_logit_error"], error)
            comparisons["source_tasks"] += 1
            adapter.set_task_context(context)
            predictor = adapter.fit((*task.support_set, task.num_ways, task.num_shots))
            adapter_probs = predictor.predict(task.query_set[0])
            operations["adapter_checks"] += 1
            script_probs = outputs["B"].softmax(1).numpy()
            np.testing.assert_allclose(adapter_probs, script_probs, atol=options["atol"], rtol=options["rtol"])
            if not np.array_equal(adapter_probs.argmax(1), script_probs.argmax(1)):
                raise AssertionError(f"Local adapter prediction mismatch: {record['task_id']}")
            comparisons["adapter_max_probability_error"] = max(comparisons["adapter_max_probability_error"],
                                                                float(np.abs(adapter_probs - script_probs).max()))
            comparisons["adapter_tasks"] += 1
            if seen:
                for expert in experiment.seen:
                    key = experiment.specialist_key(expert)
                    if expert == domain:
                        logits = outputs["B"]
                    else:
                        logits, _ = experiment.evaluate(task, key)
                        operations["off_domain_specialists"] += 1
                    correct, total = score(logits, task)
                    matrix.append(dict(task_id=record["task_id"], domain=domain, specialist=expert,
                                       correct=correct, total=total, accuracy=correct / total))
            summary["completed_tasks"] = task_index
            summary["evaluation_seconds"] = time.perf_counter() - start
    summary["phase"] = "frozen_state_checks"
    experiment.assert_fixed()
    adapter.experiment.assert_fixed()
    comparisons["reference_sampler_tasks"] = test_loader.parity_tasks
    for key, state in frozen_before.items():
        after = experiment.branch(key).state()
        for name in ("m",):
            if not torch.equal(state[name], after[name]):
                raise AssertionError(f"Evaluation mutated {key}/{name}")
        for name in state["gate_net"]:
            if not torch.equal(state["gate_net"][name], after["gate_net"][name]):
                raise AssertionError(f"Evaluation mutated {key} GateNet")
        for name in ("tasks", "steps", "pending", "initialized"):
            if state[name] != after[name]:
                raise AssertionError(f"Evaluation mutated {key}/{name}")
    matrix_summary = []
    for domain in experiment.seen:
        for expert in experiment.seen:
            selected = [row for row in matrix if row["domain"] == domain and row["specialist"] == expert]
            matrix_summary.append(dict(domain=domain, specialist=expert,
                                       accuracy=float(np.mean([r["accuracy"] for r in selected])),
                                       query_micro_accuracy=sum(r["correct"] for r in selected) / sum(r["total"] for r in selected),
                                       tasks=len(selected)))
    for filename, data in (("domain-specialist-matrix.csv", matrix_summary), ("matrix-tasks.csv", matrix)):
        with (output / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    groups = {"seen": aggregate([row for row in rows if row["seen"]]),
              "unseen": aggregate([row for row in rows if not row["seen"]]), "all": aggregate(rows)}
    seen_scores = groups["seen"]
    summary["phase"] = "aggregate_results"
    summary.update(
        description="Learning domain-specific low-rank coefficients on a frozen shared encoder and U,V",
        groups=groups,
        datasets={name: aggregate([r for r in rows if r["dataset"] == name]) for name in sorted({r["dataset"] for r in rows})},
        domains={name: aggregate([r for r in rows if r["domain"] == name]) for name in DOMAIN_INDEX},
        primary_B_minus_C=paired_bootstrap(rows, options["bootstrap_seed"], options["bootstrap_samples"]),
        auxiliary={"B_minus_" + other: seen_scores["B"]["accuracy"] - seen_scores[other]["accuracy"]
                   for other in ("A", "D")},
        diagnostics=experiment.diagnostics(), validation=experiment.validation,
        comparisons=comparisons,
        manifest_sha256=json_hash(identities), manifest_file_sha256=file_hash(output / "test-manifest.jsonl"),
        checkpoint_sha256=file_hash(args.checkpoint), source_sha256=file_hash(args.source_checkpoint),
        data_seed=options["data_seed"], test_seed=options["test_seed"], test_tasks_per_dataset=args.test_tasks_per_dataset,
        evaluation_seconds=time.perf_counter() - start, evaluation_adaptation_operations=operations,
        training=experiment.train_summary, provenance=experiment.provenance,
        tolerances={key: options[key] for key in ("atol", "rtol")},
        interpretation="C is the budget-matched primary control; D matches per-network steps only when B is divisible by meta-batch. No test checkpoint selection. No separate EMA/warmup causal claim.")
    # Persist all measurements before attempting historical score comparisons.
    summary["phase"] = "historical_comparison"
    write_summary(output / "summary.json", summary)
    try:
        summary["historical_results"] = compare_old_results(historical, rows, baseline_predictions, options["atol"])
    except Exception as error:
        summary["historical_results"] = dict(status="failed", error=error_record(error))
        raise
    if summary["historical_results"]["status"] == "failed":
        raise AssertionError("Historical task results differ; details are saved in summary.json")
    summary["phase"] = "complete"


if __name__ == "__main__":
    main()
