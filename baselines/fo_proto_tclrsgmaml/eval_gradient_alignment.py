"""Support/query gradient alignment on a saved LR-only TC checkpoint.

Custom meta-validation diagnostic, not an official benchmark score. Default:
100 total task attempts, 3 repeats, nested 1/5/10-shot supports, 5-way/20-query.
See eval_gradient_alignment.md. This module runs only when explicitly invoked.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_gradient_reliability import (
    BENEFITS, ImageLoadError, average, benefit, class_inventory, correlation,
    digest_file, frozen_condition, load_set, sample_set, write_csv,
)
from eval_inner_steps import observational_diagnostics, validate_checkpoint_config
from helpers_fo_proto_tclrsgmaml import adaptation_gamma, prototype_head
from cdmetadl.helpers.general_helpers import prepare_datasets_information
from cdmetadl.ingestion.image_dataset import create_datasets
from model import MyLearner

WAYS, QUERY_COUNT, SUPPORT_POOL, INNER_STEPS = 5, 20, 10, 5
DIRECTIONS = ("raw", "base", "full")
NORMS = tuple("norm_" + name for name in (*DIRECTIONS, "query"))
DOTS = tuple("dot_" + name + "_query" for name in DIRECTIONS)
COSINES = tuple("cos_" + name for name in DIRECTIONS)
DELTAS = ("delta_cos_raw", "delta_cos_base")
ALIGNMENT = NORMS + DOTS + COSINES + DELTAS
STATUSES = tuple(key + "_status" for key in NORMS + COSINES + DELTAS)
START_LOSSES = ("support_ce_start", "query_ce_start")
METRICS = ALIGNMENT + START_LOSSES + BENEFITS
IDS = ("task_id", "dataset", "shot", "repeat_id")
REPEAT_FIELDS = IDS + ("encoder_tensors", "low_rank_tensors") + METRICS + STATUSES
TENSOR_FIELDS = IDS + ("tensor", "shape", "has_low_rank") + ALIGNMENT + STATUSES


def finite(value):
    return float(value) if math.isfinite(value) else None


def metrics_from_moments(moments, min_norm):
    """Moments: raw/base/full/query squared norms, then three query dots."""
    values = moments.detach().cpu().tolist()
    result = {}
    for key, squared in zip(NORMS, values[:4], strict=True):
        norm = math.sqrt(squared) if math.isfinite(squared) and squared >= 0 else None
        result[key] = norm
        result[key + "_status"] = ("nonfinite" if norm is None else
                                   "zero_or_too_small" if norm <= min_norm else "valid")
    result.update({key: finite(dot) for key, dot in zip(DOTS, values[4:], strict=True)})
    for name in DIRECTIONS:
        key = "cos_" + name
        issues = [norm + ":" + result[norm + "_status"] for norm in ("norm_" + name, "norm_query")
                  if result[norm + "_status"] != "valid"]
        dot = result["dot_" + name + "_query"]
        result[key] = None
        if issues:
            status = ";".join(issues)
        elif dot is None:
            status = "nonfinite_dot"
        else:
            value = dot / result["norm_" + name] / result["norm_query"]
            if not math.isfinite(value) or abs(value) > 1 + 1e-12:
                status = "nonfinite_or_out_of_range"
            else:
                result[key] = max(-1.0, min(1.0, value))  # rounding only; no norm epsilon
                status = "valid"
        result[key + "_status"] = status
    for reference in ("raw", "base"):
        key = "delta_cos_" + reference
        valid = result["cos_full"] is not None and result["cos_" + reference] is not None
        result[key] = result["cos_full"] - result["cos_" + reference] if valid else None
        result[key + "_status"] = "valid" if valid else "invalid_input_cosine"
    return result


@torch.enable_grad()
def initial_gradients(learner, support, labels, query, query_labels):
    """Both losses at identical initial coordinates; query is never clipped."""
    model = learner.learner
    body = [p.detach().clone().requires_grad_(True) for p in model.parameters()]
    with torch.no_grad():
        features = model.forward_weights(support, body, embedding=True)
        head = prototype_head(features, labels, WAYS)
        condition = frozen_condition(learner.transport, features)
    head = [p.detach().clone().requires_grad_(True) for p in head]
    coordinates = body + head
    support_loss = F.cross_entropy(model.forward_weights(support, coordinates), labels)
    gs = torch.autograd.grad(support_loss, coordinates, create_graph=False)[:len(body)]
    support_ce = support_loss.item()
    del support_loss
    # Only a new forward graph: coordinates/head have not been updated.
    query_loss = F.cross_entropy(model.forward_weights(query, coordinates), query_labels)
    gq = torch.autograd.grad(query_loss, body, create_graph=False)
    clip = learner.config["method_config"]["grad_clip"]
    gs = [g.detach() if clip is None else g.detach().clamp(-clip, clip) for g in gs]
    return gs, [g.detach() for g in gq], condition, dict(
        support_ce_start=finite(support_ce), query_ce_start=finite(query_loss.item()))


@torch.no_grad()
def alignment(transport, gs, gq, condition, min_norm, *, gamma):
    # Same residual-off mechanism as reliability.ResidualControl: only (1+c)=0.
    base_condition = (condition[0], {key: -torch.ones_like(value) for key, value in condition[1].items()})
    totals = torch.zeros(7, dtype=torch.float64, device=gs[0].device)
    rows = []
    for name, support_grad, query_grad in zip(transport.names, gs, gq, strict=True):
        base = transport.transport_gradient(name, support_grad, base_condition, gamma=gamma)
        full = transport.transport_gradient(name, support_grad, condition, gamma=gamma)
        directions = [support_grad.double(), base.double(), full.double()]
        q = query_grad.double()
        moments = torch.stack([g.square().sum() for g in directions] + [q.square().sum()]
                              + [(g * q).sum() for g in directions])
        # Accumulate ALL coordinates, including tensors with individually invalid
        # cosines. Never average tensor cosines or silently drop their coordinates.
        totals.add_(moments)
        rows.append(dict(tensor=name, shape=json.dumps(list(support_grad.shape)),
                         has_low_rank=transport.indices[name] in transport.u,
                         **metrics_from_moments(moments, min_norm)))
    return metrics_from_moments(totals, min_norm), rows


def make_manifest(datasets, pools, num_tasks, num_repeats, shots, seed):
    task_rng = np.random.RandomState(seed)
    result = []
    for task_id in range(num_tasks):
        dataset_index = int(task_rng.randint(len(datasets)))
        dataset, pool = datasets[dataset_index], pools[dataset_index]
        task = dict(task_id=task_id, dataset=dataset.name, dataset_index=dataset_index,
                    eligible_classes=len(pool), status="planned", skip_reason=None)
        result.append(task)
        if len(pool) < WAYS:
            task.update(status="skipped", skip_reason="fewer_than_5_classes_with_30_usable_images")
            continue
        classes = task_rng.choice(list(pool), WAYS, replace=False).tolist()
        query_by_class = [task_rng.choice(pool[c], QUERY_COUNT, replace=False).tolist() for c in classes]
        query = sample_set(query_by_class, task_rng)
        query["paths"] = [dataset.img_paths[i] for i in query["indices"]]
        remaining = [np.setdiff1d(pool[c], q) for c, q in zip(classes, query_by_class, strict=True)]
        task.update(class_ids=classes, query=query, repeats=[])
        for repeat_id in range(num_repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, task_id, repeat_id, 2]))
            ordered = [rng.choice(indices, SUPPORT_POOL, replace=False).tolist() for indices in remaining]
            largest = sample_set(ordered, rng)
            largest["paths"] = [dataset.img_paths[i] for i in largest["indices"]]
            if set(largest["indices"]) & set(query["indices"]):
                raise ValueError("Support/query sample overlap")
            subsets = {}
            previous = set()
            for shot in shots:
                keep = {i for selected in ordered for i in selected[:shot]}
                positions = [j for j, index in enumerate(largest["indices"]) if index in keep]
                subset = {key: [largest[key][j] for j in positions] for key in ("indices", "labels", "paths")}
                subset["positions_in_support10"] = positions
                if (not previous.issubset(keep) or len(positions) != WAYS * shot
                        or Counter(subset["labels"]) != Counter({i: shot for i in range(WAYS)})):
                    raise ValueError("Invalid nested support sample")
                previous = keep
                subsets[str(shot)] = subset
            task["repeats"].append(dict(repeat_id=repeat_id, ordered_support_indices_by_class=ordered,
                                         support10=largest, shots=subsets))
    return result


def evaluate_task(learner, dataset, task, shots, min_norm):
    gamma = adaptation_gamma(learner.config["method_config"], "validation")
    # Preload all selected images so I/O errors exclude the whole task, all shots.
    query, query_labels = load_set(dataset, task["query"], task["class_ids"])
    supports = [load_set(dataset, repeat["support10"], task["class_ids"]) for repeat in task["repeats"]]
    query, query_labels = query.to(learner.dev), query_labels.to(learner.dev)
    repeat_rows, tensor_rows = [], []
    for repeat, (images, labels) in zip(task["repeats"], supports, strict=True):
        for shot in shots:
            positions = repeat["shots"][str(shot)]["positions_in_support10"]
            support, sy = images[positions].to(learner.dev), labels[positions].to(learner.dev)
            ids = dict(task_id=task["task_id"], dataset=task["dataset"], shot=shot, repeat_id=repeat["repeat_id"])
            gs, gq, condition, losses = initial_gradients(learner, support, sy, query, query_labels)
            whole, tensors = alignment(learner.transport, gs, gq, condition, min_norm, gamma=gamma)
            del gs, gq, condition
            # Each evaluation starts from checkpoint coordinates again, through
            # unchanged adapt(); the measured query gradient never enters updates.
            gains = benefit(learner, support, sy, query, query_labels, gamma=gamma)
            repeat_rows.append(dict(ids, encoder_tensors=len(tensors),
                low_rank_tensors=sum(row["has_low_rank"] for row in tensors), **whole, **losses, **gains))
            tensor_rows.extend(dict(ids, **row) for row in tensors)
    return repeat_rows, tensor_rows


def aggregate_tasks(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task_id"], row["dataset"], row["shot"])].append(row)
    return [dict(task_id=task_id, dataset=dataset, shot=shot, num_repeats=len(group),
                 **{key: average([r[key] for r in group]) for key in METRICS},
                 **{key + "_valid_repeats": sum(r[key] is not None for r in group) for key in METRICS})
            for (task_id, dataset, shot), group in groups.items()]


def association(rows, x, y):
    # Use matching valid repeats for both x and gain before computing one
    # observation per task. No pseudoreplication or unequal support subsets.
    groups = defaultdict(list)
    for row in rows:
        if row[x] is not None and row[y] is not None:
            groups[row["task_id"]].append(row)
    paired = [{x: average([r[x] for r in group]), y: average([r[y] for r in group])}
              for group in groups.values()]
    return dict(correlation(paired, x, y), valid_repeats=sum(map(len, groups.values())))


def paired_shots(rows, shots):
    indexed = {(r["task_id"], r["shot"], r["repeat_id"]): r for r in rows}
    by_task_shot = defaultdict(list)
    for row in rows:
        by_task_shot[(row["task_id"], row["shot"])].append(row)
    result = []
    task_ids = sorted({r["task_id"] for r in rows})
    for lower, higher in combinations(shots, 2):
        metrics = {}
        for key in METRICS:
            task_differences, valid_repeats, valid_ids = [], 0, []
            for task_id in task_ids:
                differences = []
                for row in by_task_shot[(task_id, lower)]:
                    other = indexed.get((task_id, higher, row["repeat_id"]))
                    if other is not None and row[key] is not None and other[key] is not None:
                        differences.append(other[key] - row[key])
                if differences:
                    task_differences.append(average(differences))
                    valid_repeats += len(differences)
                    valid_ids.append(task_id)
            metrics[key] = dict(mean_higher_minus_lower=average(task_differences),
                                valid_tasks=len(task_differences), valid_repeats=valid_repeats,
                                task_ids=valid_ids)
        result.append(dict(lower_shot=lower, higher_shot=higher, metrics=metrics))
    return result


def summarize(rows, task_means, tensor_counts, manifest, inventory, shots):
    summary = dict(dataset_shot=[], paired_shots_by_dataset={}, datasets=inventory,
        aggregation="Valid repeats averaged within task first; task means weighted equally. "
                    "Shot differences pair identical task AND repeat, using common-valid values. "
                    "Correlations pair valid repeats before task averaging, within dataset/shot. "
                    "Descriptive associations only; no causal inference.",
        invalid_tensor_measurements=[dict(dataset=d, shot=s, tensor=t, field=f, status=status, count=count)
                                    for (d, s, t, f, status), count in sorted(tensor_counts.items())])
    for report in inventory:
        dataset = report["dataset"]
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        summary["paired_shots_by_dataset"][dataset] = paired_shots(dataset_rows, shots)
        for shot in shots:
            group = [r for r in dataset_rows if r["shot"] == shot]
            means = [r for r in task_means if r["dataset"] == dataset and r["shot"] == shot]
            summary["dataset_shot"].append(dict(dataset=dataset, shot=shot,
                attempted_tasks=sum(t["dataset"] == dataset for t in manifest),
                completed_tasks=len(means), repeat_measurements=len(group),
                **{key: average([r[key] for r in means]) for key in METRICS},
                valid_tasks={key: sum(r[key] is not None for r in means) for key in METRICS},
                valid_repeats={key: sum(r[key] is not None for r in group) for key in METRICS},
                status_counts={key: dict(Counter(r[key] for r in group)) for key in STATUSES},
                correlations={x + "_vs_" + y: association(group, x, y)
                              for x in DELTAS for y in ("accuracy_gain", "loss_gain")}))
    summary["skipped_tasks"] = [dict(task_id=t["task_id"], dataset=t["dataset"], reason=t["skip_reason"])
                                for t in manifest if t["status"] == "skipped"]
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--data_seed", type=int, required=True, help="Checkpoint training run's data seed")
    parser.add_argument("--sampling_seed", type=int, default=12345)
    parser.add_argument("--num_tasks", type=int, default=100, help="TOTAL attempted class selections")
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--output_dir", required=True, help="New directory; existing directories rejected")
    parser.add_argument("--min_norm", type=float, default=1e-10, help="L2 norm floor; no epsilon is added")
    args = parser.parse_args(argv)
    if any(not 0 <= seed < 2**32 for seed in (args.data_seed, args.sampling_seed)):
        parser.error("Seeds must be uint32")
    if min(args.num_tasks, args.num_repeats, args.image_size) < 1:
        parser.error("Task/repeat counts and image size must be positive")
    if len(set(args.shots)) != len(args.shots) or any(not 1 <= shot <= SUPPORT_POOL for shot in args.shots):
        parser.error("--shots must be distinct integers from 1 through 10")
    if not math.isfinite(args.min_norm) or args.min_norm <= 0:
        parser.error("--min_norm must be positive and finite")
    args.shots = sorted(args.shots)
    output = Path(args.output_dir).resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new path")
    learner = MyLearner()
    learner.load(args.checkpoint)
    validate_checkpoint_config(learner.config)
    if learner.transport.names != [name for name, _ in learner.learner.named_parameters()]:
        raise ValueError("Encoder/transport parameter ordering mismatch")
    saved_seed = {}
    for name, cfg in (("config", learner.config), ("experiment_config", learner.config.get("experiment_config", {}))):
        if "data_seed" in cfg:
            saved_seed[name] = cfg["data_seed"]
            if type(cfg["data_seed"]) is not int or cfg["data_seed"] != args.data_seed:
                raise ValueError(f"CLI data seed does not match saved {name}.data_seed")
    original_config = copy.deepcopy(learner.config)
    _, valid_info, _ = prepare_datasets_information(
        args.input_data_dir, learner.config["validation_datasets"], args.data_seed, False)
    datasets = create_datasets(valid_info, args.image_size)
    if not datasets:
        raise ValueError("No meta-validation datasets")
    # Reliability also requires 30 images: its 20+5+5 equals our 20+10.
    inventories = [class_inventory(dataset) for dataset in datasets]
    pools, inventory = [r[0] for r in inventories], [r[1] for r in inventories]
    manifest = make_manifest(datasets, pools, args.num_tasks, args.num_repeats, args.shots, args.sampling_seed)
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "max-va.pth"
    metadata = dict(status="running", cli=vars(args), checkpoint=str(checkpoint),
        checkpoint_sha256=digest_file(checkpoint), checkpoint_config=original_config,
        eval_gamma=adaptation_gamma(learner.config["method_config"], "validation"),
        architecture=learner.transport.architecture(), best_validation_accuracy=learner.best_score,
        device=str(learner.dev), torch_version=str(torch.__version__), numpy_version=np.__version__,
        data_seed_source="checked against saved config" if saved_seed else "user supplied from training log; not saved in checkpoint",
        saved_seed_evidence=saved_seed, split="meta-validation", index_base=0,
        sample_index_definition="ImageDataset index / zero-based labels.csv row",
        validation_datasets={name: dict(info=list(info), labels_sha256=digest_file(info[3])) for name, info in valid_info.items()},
        split_file_sha256=digest_file(Path(args.input_data_dir) / "info/meta_splits.txt"),
        protocol=dict(ways=WAYS, query_per_class=QUERY_COUNT, support_pool_per_class=SUPPORT_POOL,
            shots=args.shots, inner_steps=INNER_STEPS, total_attempts=args.num_tasks, repeats=args.num_repeats,
            uniform_dataset_then_eligible_classes=True, fixed_classes_query=True, nested_supports=True,
            support_query_disjoint=True, across_repeat_overlap_allowed=True, replace_skipped_tasks=False,
            minimum_usable_images_per_class=30, missing_and_duplicate_paths_excluded=True,
            official_benchmark=False),
        gradient_definition="At identical initial encoder and support-prototype head: independent leaf coordinates; "
            "support encoder gradient elementwise clipped per saved config; query gradient never clipped. "
            "No differentiation through prototypes into encoder. Query gradient is diagnostic only.",
        batch_statistics="Unchanged functional BN: training=True with new local buffers on each forward. "
            "Support and query use their own batch statistics; prototype forward uses support statistics. "
            "Shot effects include support batch-statistics and prototype changes. Query is one complete batch.",
        numerical_policy=dict(min_norm=args.min_norm, units="L2 norm", accumulation="float64 norms/dots",
            transport_dtype="checkpoint dtype", encoder_cosine="summed coordinate dots/squared norms, never mean of tensor cosines",
            epsilon_added=False, invalid_csv="blank plus status", invalid_json=None,
            cosine_rounding="clamp only finite values within 1e-12 of [-1,1]; otherwise invalid"),
        gain_signs=dict(accuracy_gain="on minus off", loss_gain="CE off minus CE on"),
        state_checks=dict(parameters_buffers="not_completed", config="not_completed", persistent_grads="not_completed"),
        source_sha256={str(path): digest_file(ROOT / path) for path in [
            "baselines/fo_proto_tclrsgmaml/" + name for name in (Path(__file__).name,
            "eval_gradient_reliability.py", "eval_inner_steps.py", "helpers_fo_proto_tclrsgmaml.py",
            "model.py", "network.py", "task_transport.py", "low_rank_transport.py")]
            + ["cdmetadl/ingestion/image_dataset.py", "cdmetadl/helpers/general_helpers.py"]})
    output.mkdir(parents=True, exist_ok=False)
    all_rows, tensor_counts = [], Counter()
    with ExitStack() as stack:
        handles = {name: stack.enter_context((output / name).open("x", encoding="utf-8", newline=""))
                   for name in ("alignment_repeats.csv", "alignment_tensors.csv", "task_shot_means.csv",
                                "sampling_manifest.json", "metadata.json", "summary.json")}
        repeat_writer = write_csv(handles["alignment_repeats.csv"], REPEAT_FIELDS, [])
        tensor_writer = write_csv(handles["alignment_tensors.csv"], TENSOR_FIELDS, [])
        try:
            with observational_diagnostics(learner):
                for task in manifest:
                    if task["status"] == "skipped":
                        continue
                    task["status"] = "running"
                    try:
                        rows, tensors = evaluate_task(learner, datasets[task["dataset_index"]], task,
                                                     args.shots, args.min_norm)
                    except ImageLoadError as exc:
                        task.update(status="skipped", skip_reason=str(exc))
                        continue
                    repeat_writer.writerows(rows)
                    tensor_writer.writerows(tensors)
                    handles["alignment_repeats.csv"].flush()
                    handles["alignment_tensors.csv"].flush()
                    all_rows.extend(rows)
                    for row in tensors:
                        for field in STATUSES:
                            if row[field] != "valid":
                                tensor_counts[(row["dataset"], row["shot"], row["tensor"], field, row[field])] += 1
                    task["status"] = "completed"
                    print(f"Task {task['task_id'] + 1}/{args.num_tasks}: {task['dataset']}, all shots/repeats complete", flush=True)
                if learner.config != original_config:
                    metadata["state_checks"]["config"] = "failed"
                    raise RuntimeError("Checkpoint config changed")
                metadata["state_checks"]["config"] = "passed"
                if any(p.grad is not None for root in (learner.learner, learner.transport) for p in root.parameters()):
                    metadata["state_checks"]["persistent_grads"] = "failed"
                    raise RuntimeError("Persistent parameter gradients accumulated")
                metadata["state_checks"]["persistent_grads"] = "passed"
            metadata["state_checks"]["parameters_buffers"] = "passed"
            metadata["status"] = "completed" if all_rows else "no_completed_tasks"
        except BaseException as exc:
            metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if str(exc).startswith("Diagnostic mutated parameter/buffer:"):
                metadata["state_checks"]["parameters_buffers"] = "failed"
            raise
        finally:
            for task in manifest:
                if task["status"] == "running":
                    task["status"] = "failed"
            metadata["task_status_counts"] = dict(Counter(t["status"] for t in manifest))
            task_means = aggregate_tasks(all_rows)
            fields = ("task_id", "dataset", "shot", "num_repeats") + METRICS + tuple(
                key + "_valid_repeats" for key in METRICS)
            write_csv(handles["task_shot_means.csv"], fields, task_means)
            summary = summarize(all_rows, task_means, tensor_counts, manifest, inventory, args.shots)
            summary.update(run_status=metadata["status"], state_checks=metadata["state_checks"], min_norm=args.min_norm)
            for name, value in (("metadata.json", metadata), ("summary.json", summary),
                                ("sampling_manifest.json", dict(class_inventory=inventory, tasks=manifest))):
                json.dump(value, handles[name], indent=2, allow_nan=False)
                handles[name].write("\n")
    for row in summary["dataset_shot"]:
        print(f"{row['dataset']} shot={row['shot']}: n_tasks={row['completed_tasks']}, "
              f"delta_cos_base={row['delta_cos_base']}, accuracy_gain={row['accuracy_gain']}, "
              f"valid delta repeats={row['valid_repeats']['delta_cos_base']}/{row['repeat_measurements']}")
    print(f"Custom diagnostic outputs: {output}; status={metadata['status']}")


if __name__ == "__main__":
    main()
