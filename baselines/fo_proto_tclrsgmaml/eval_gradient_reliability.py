"""Support-sampling diagnostics on meta-validation; not a benchmark score.

Loads max-va.pth through MyLearner.load. No optimizer or outer loss is created.
Default: 100 task attempts TOTAL, uniform dataset selection, 3 support pairs per
class selection. Classes/query stay fixed across pairs; A/B/query are disjoint
within each pair. Different pairs may reuse support images. See the companion
eval_gradient_reliability.md for definitions, exclusions and invocation.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
import csv
import hashlib
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

from helpers_fo_proto_tclrsgmaml import adapt, prototype_head
from eval_inner_steps import validate_checkpoint_config, observational_diagnostics
from cdmetadl.helpers.general_helpers import prepare_datasets_information
from cdmetadl.ingestion.image_dataset import create_datasets
from model import MyLearner

WAYS, SHOTS, QUERIES, STEPS = 5, 5, 20, 5
ENERGY_FIELDS = ("raw_disagreement_sq", "raw_mean_sq", "fixed_disagreement_sq",
                 "fixed_mean_sq", "system_difference_sq", "system_disagreement_sq", "system_mean_sq")
MEASURES = ("r0", "rP", "rP_over_r0", "r_system")
LAYER_FIELDS = ("task_id", "dataset", "pair_id", "layer", "shape", "has_low_rank",
                "fixed_condition") + ENERGY_FIELDS + MEASURES + tuple(
                    name + "_status" for name in MEASURES)
BENEFITS = ("accuracy_lr_on", "accuracy_lr_off", "accuracy_gain",
            "query_ce_lr_on", "query_ce_lr_off", "loss_gain")
PAIR_FIELDS = ("task_id", "dataset", "pair_id", "support") + BENEFITS
TASK_FIELDS = ("task_id", "dataset", "num_pairs", "num_support_evaluations") + BENEFITS


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return math.fsum(v / len(values) for v in values) if values else None


def ratio(numerator, denominator, floor):
    if numerator is None or denominator is None:
        return None, "nonfinite_energy"
    if denominator <= floor:
        return None, "denominator_zero_or_too_small"
    value = numerator / denominator
    return (value, "valid") if math.isfinite(value) else (None, "nonfinite_ratio")


def energy(tensor):
    value = tensor.detach().double().square().sum().item()
    return value if math.isfinite(value) else None


def frozen_condition(transport, features):
    with torch.no_grad():
        condition = transport.condition(features.mean(0).detach())
        if condition is None or not bool((condition[0] == 0).all()):
            raise ValueError("LR-only TC requires finite, zero scalar deltas")
        a, c = condition
        if not all(bool(torch.isfinite(v).all()) for v in [a, *c.values()]):
            raise ValueError("Nonfinite support-conditioned coefficients")
        return a.detach().clone(), {k: v.detach().clone() for k, v in c.items()}


@torch.enable_grad()
def first_gradients(learner, support, labels):
    """First-step encoder derivatives with independent encoder/head coordinates."""
    model = learner.learner
    # No persistent parameter participates in the measurement graph.
    body = [w.detach().clone().requires_grad_(True) for w in model.parameters()]
    with torch.no_grad():
        features = model.forward_weights(support, body, embedding=True)
        head = prototype_head(features, labels, WAYS)
        condition = frozen_condition(learner.transport, features)
    # Treat W,b as independent leaves: no prototype-to-encoder inner derivative.
    head = [w.detach().clone().requires_grad_(True) for w in head]
    loss = F.cross_entropy(model.forward_weights(support, body + head), labels)
    gradients = torch.autograd.grad(loss, body + head, create_graph=False)
    clip = learner.config["method_config"]["grad_clip"]
    gradients = [g.detach() if clip is None else g.detach().clamp(-clip, clip)
                 for g in gradients]
    return gradients[:len(body)], condition


class ResidualControl:
    """Read-only view for unchanged adapt(): learned scalar is always retained."""
    def __init__(self, transport, residual_on):
        self.transport = transport
        self.names = transport.names
        self.residual_on = residual_on

    @torch.no_grad()
    def condition(self, embedding):
        result = self.transport.condition(embedding)
        if result is None or not bool((result[0] == 0).all()):
            raise ValueError("Expected LR-only TC conditioning")
        a, c = result
        if not self.residual_on:
            # Existing transport multiplies its rank projection by (1+delta_c).
            # -1 zeros ONLY that residual, without editing beta, U/V or config.
            c = {key: -torch.ones_like(value) for key, value in c.items()}
        return a.detach(), {key: value.detach() for key, value in c.items()}

    @torch.no_grad()
    def transport_gradient(self, name, gradient, conditioning, *, gamma=1.0):
        return self.transport.transport_gradient(name, gradient, conditioning, gamma=gamma)


def benefit(learner, support, labels, query, query_labels):
    scores = {}
    cfg = dict(learner.config["method_config"], inner_steps=STEPS)
    for mode, residual_on in (("lr_on", True), ("lr_off", False)):
        weights = [w.detach().clone().requires_grad_(True)
                   for w in learner.learner.parameters()]
        fast = adapt(learner.learner, weights, support, labels, cfg, WAYS,
                     ResidualControl(learner.transport, residual_on), phase="test")
        with torch.no_grad():
            logits = learner.learner.forward_weights(query, fast)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError(f"Nonfinite {mode} query logits")
            scores[f"accuracy_{mode}"] = (logits.argmax(1) == query_labels).double().mean().item()
            scores[f"query_ce_{mode}"] = F.cross_entropy(logits, query_labels).item()
        del fast, weights
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite query score")
    scores["accuracy_gain"] = scores["accuracy_lr_on"] - scores["accuracy_lr_off"]
    scores["loss_gain"] = scores["query_ce_lr_off"] - scores["query_ce_lr_on"]
    return scores


@torch.no_grad()
def pair_measurements(transport, ga, gb, ca, cb, floor):
    rows = []
    for name, a, b in zip(transport.names, ga, gb, strict=True):
        mean, disagreement = (a + b) / 2, (a - b) / 2
        raw_d, raw_m = energy(disagreement), energy(mean)
        # This separately measures the complete support-dependent system.
        pa = transport.transport_gradient(name, a, ca)
        pb = transport.transport_gradient(name, b, cb)
        system_d = energy((pa.double() - pb.double()) / 2)
        system_m = energy((pa.double() + pb.double()) / 2)
        system_difference = energy(pa.double() - pb.double())
        r0, r0_status = ratio(raw_d, raw_m, floor)
        rs, rs_status = ratio(system_d, system_m, floor)
        for label, condition in (("A", ca), ("B", cb)):
            # Frozen P applied to BOTH coordinates; no dense P is constructed.
            pd = transport.transport_gradient(name, disagreement, condition)
            pm = transport.transport_gradient(name, mean, condition)
            fixed_d, fixed_m = energy(pd), energy(pm)
            rp, rp_status = ratio(fixed_d, fixed_m, floor)
            amplification, amp_status = None, "invalid_r0_or_rP"
            if r0 is not None and rp is not None:
                if raw_d <= floor or r0 == 0:
                    amp_status = "baseline_disagreement_zero_or_too_small"
                else:
                    amplification, amp_status = ratio(rp, r0, 0.0)
            rows.append(dict(layer=name, shape=json.dumps(list(a.shape)),
                has_low_rank=transport.indices[name] in transport.u, fixed_condition=label,
                raw_disagreement_sq=raw_d, raw_mean_sq=raw_m,
                fixed_disagreement_sq=fixed_d, fixed_mean_sq=fixed_m,
                system_difference_sq=system_difference,
                system_disagreement_sq=system_d, system_mean_sq=system_m,
                r0=r0, rP=rp, rP_over_r0=amplification, r_system=rs,
                r0_status=r0_status, rP_status=rp_status,
                rP_over_r0_status=amp_status, r_system_status=rs_status))
    return rows


def class_inventory(dataset):
    """Exclude duplicate file paths so index-disjoint also means path-disjoint."""
    paths = [str(Path(p).resolve()) for p in dataset.img_paths]
    counts = Counter(paths)
    exists = {path: Path(path).is_file() for path in counts}
    pools, classes = {}, []
    for class_id, indices in enumerate(dataset.idx_per_label):
        usable = [int(i) for i in indices if counts[paths[int(i)]] == 1 and exists[paths[int(i)]]]
        eligible = len(usable) >= QUERIES + 2 * SHOTS
        classes.append(dict(class_id=class_id, raw_images=len(indices),
            usable_images=len(usable), excluded_rows=len(indices) - len(usable),
            duplicate_path_rows=sum(counts[paths[int(i)]] > 1 for i in indices),
            missing_file_rows=sum(not exists[paths[int(i)]] for i in indices),
            eligible=eligible, exclusion_reason=None if eligible else "fewer_than_30_unique_existing_path_images"))
        if eligible:
            pools[class_id] = np.asarray(usable, dtype=np.int64)
    return pools, dict(dataset=dataset.name, total_classes=len(classes),
                       eligible_classes=len(pools), classes=classes)


def sample_set(by_class, rng):
    indices = np.asarray(by_class).T.reshape(-1)
    labels = np.tile(np.arange(WAYS), len(by_class[0]))
    order = rng.permutation(len(indices))
    return dict(indices=indices[order].tolist(), labels=labels[order].tolist())


def make_manifest(datasets, pools, tasks, repeats, seed):
    # Separate RNG streams make class/query selections stable if repeats changes.
    task_rng = np.random.RandomState(seed)
    manifest = []
    for task_id in range(tasks):
        dataset_index = int(task_rng.randint(len(datasets)))
        dataset, pool = datasets[dataset_index], pools[dataset_index]
        row = dict(task_id=task_id, dataset=dataset.name, dataset_index=dataset_index,
                   eligible_classes=len(pool), status="planned", skip_reason=None)
        manifest.append(row)
        if len(pool) < WAYS:
            row.update(status="skipped", skip_reason="fewer_than_5_eligible_classes")
            continue
        classes = task_rng.choice(list(pool), WAYS, replace=False).tolist()
        query = [task_rng.choice(pool[c], QUERIES, replace=False).tolist() for c in classes]
        remaining = [np.setdiff1d(pool[c], q) for c, q in zip(classes, query)]
        row.update(class_ids=classes, query=sample_set(query, task_rng), pairs=[])
        for pair_id in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, task_id, pair_id, 1]))
            choices = [rng.choice(indices, 2 * SHOTS, replace=False) for indices in remaining]
            a = sample_set([x[:SHOTS] for x in choices], rng)
            b = sample_set([x[SHOTS:] for x in choices], rng)
            if (set(a["indices"]) & set(b["indices"]) or
                    (set(a["indices"]) | set(b["indices"])) & set(row["query"]["indices"])):
                raise ValueError("Sampler produced overlapping A/B/query indices")
            row["pairs"].append(dict(pair_id=pair_id, A=a, B=b))
        for selected in [row["query"], *[p[s] for p in row["pairs"] for s in ("A", "B")]]:
            selected["paths"] = [dataset.img_paths[i] for i in selected["indices"]]
    return manifest


class ImageLoadError(RuntimeError):
    pass


def load_set(dataset, selected, classes):
    images = []
    for index, label in zip(selected["indices"], selected["labels"], strict=True):
        try:
            image, original = dataset[index]
        except (OSError, ValueError, RuntimeError) as exc:
            raise ImageLoadError(f"image index {index}: {exc}") from exc
        if int(original) != classes[label]:
            raise ValueError("Manifest labels differ from dataset labels")
        if image.ndim != 3 or image.shape[0] != 3:
            raise ImageLoadError(f"image index {index}: expected RGB, got {tuple(image.shape)}")
        images.append(image)
    return torch.stack(images), torch.tensor(selected["labels"], dtype=torch.long)


def evaluate_task(learner, dataset, task, floor):
    # Load every image before computing: a read failure excludes the entire task.
    q, qy = load_set(dataset, task["query"], task["class_ids"])
    supports = [(pair, {s: load_set(dataset, pair[s], task["class_ids"]) for s in ("A", "B")})
                for pair in task["pairs"]]
    q, qy = q.to(learner.dev), qy.to(learner.dev)
    layer_rows, score_rows = [], []
    for pair, sets in supports:
        common = dict(task_id=task["task_id"], dataset=task["dataset"], pair_id=pair["pair_id"])
        gradients, conditions = {}, {}
        for s in ("A", "B"):
            x, y = (t.to(learner.dev) for t in sets[s])
            gradients[s], conditions[s] = first_gradients(learner, x, y)
            score_rows.append(dict(common, support=s, **benefit(learner, x, y, q, qy)))
        layer_rows.extend(dict(common, **row) for row in pair_measurements(
            learner.transport, gradients["A"], gradients["B"], conditions["A"], conditions["B"], floor))
    task_row = dict(task_id=task["task_id"], dataset=task["dataset"],
                    num_pairs=len(task["pairs"]), num_support_evaluations=len(score_rows),
                    **{key: average([r[key] for r in score_rows]) for key in BENEFITS})
    return layer_rows, score_rows, task_row


def task_layer_means(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["layer"], row["fixed_condition"])].append(row)
    return [dict(task_id=group[0]["task_id"], dataset=group[0]["dataset"],
                 layer=layer, fixed_condition=condition, repeats=len(group),
                 **{key: average([r[key] for r in group]) for key in MEASURES},
                 **{key + "_valid_pairs": sum(r[key] is not None for r in group) for key in MEASURES})
            for (layer, condition), group in groups.items()]


def correlation(rows, x_key, y_key):
    pairs = [(r[x_key], r[y_key]) for r in rows if r[x_key] is not None and r[y_key] is not None]
    result = dict(n_tasks=len(pairs), pearson=None, status="fewer_than_3_valid_tasks")
    if len(pairs) < 3:
        return result
    x, y = np.asarray(pairs, dtype=np.float64).T
    # Scale before centering to prevent overflow for large but finite ratios.
    if np.ptp(x / max(np.max(np.abs(x)), 1.0)) == 0 or np.ptp(y) == 0:
        result["status"] = "constant_variable"
        return result
    x = x / max(np.max(np.abs(x)), 1.0)
    y = y / max(np.max(np.abs(y)), 1.0)
    value = float(np.corrcoef(x, y)[0, 1])
    result.update(pearson=value if math.isfinite(value) else None,
                  status="valid" if math.isfinite(value) else "nonfinite_correlation")
    return result


def summarize(tasks, layers, manifest, inventory):
    result = dict(by_dataset={}, by_dataset_layer=[],
                  interpretation="Descriptive within-dataset task-level correlations; no causality. "
                  "Support repeats are averaged within task, never independent observations.")
    for report in inventory:
        name = report["dataset"]
        group = [r for r in tasks if r["dataset"] == name]
        attempted = [r for r in manifest if r["dataset"] == name]
        result["by_dataset"][name] = dict(eligible_classes=report["eligible_classes"],
            total_classes=report["total_classes"], attempts=len(attempted), completed_tasks=len(group),
            skipped_reasons=dict(Counter(r["skip_reason"] for r in attempted if r["status"] == "skipped")),
            **{key: average([r[key] for r in group]) for key in BENEFITS})
    scores = {r["task_id"]: r for r in tasks}
    groups = defaultdict(list)
    for row in layers:
        groups[(row["dataset"], row["layer"], row["fixed_condition"])].append(row)
    for (dataset, layer, condition), group in groups.items():
        joined = [dict(r, accuracy_gain=scores[r["task_id"]]["accuracy_gain"],
                       loss_gain=scores[r["task_id"]]["loss_gain"]) for r in group]
        result["by_dataset_layer"].append(dict(dataset=dataset, layer=layer,
            fixed_condition=condition, n_tasks=len(group),
            **{key: average([r[key] for r in group]) for key in MEASURES},
            **{key + "_valid_tasks": sum(r[key] is not None for r in group) for key in MEASURES},
            **{key + "_valid_pairs": sum(r[key + "_valid_pairs"] for r in group) for key in MEASURES},
            correlations={f"{x}_vs_{y}": correlation(joined, x, y)
                          for x in ("rP_over_r0", "r_system") for y in ("accuracy_gain", "loss_gain")}))
    return result


def write_csv(handle, fields, rows):
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return writer


def summary_text(summary):
    lines = ["Custom meta-validation support-sampling diagnostic; NOT an official benchmark result.",
             f"Run status: {summary.get('run_status', 'unspecified')}",
             "Accuracy/gains are fractions. Positive accuracy_gain and loss_gain favor low-rank ON.",
             "Means weight tasks equally; repeats are aggregated within task.",
             "Correlations are descriptive and within dataset; no causal interpretation.", ""]
    for dataset, row in summary["by_dataset"].items():
        lines.append(f"{dataset}: {row['completed_tasks']}/{row['attempts']} completed tasks; "
                     f"eligible classes={row['eligible_classes']}/{row['total_classes']}; "
                     f"accuracy_gain={row['accuracy_gain']}, loss_gain={row['loss_gain']}; "
                     f"skips={row['skipped_reasons']}")
    lines.append("\nDataset / tensor / frozen P: mean task ratios and valid task counts")
    for row in summary["by_dataset_layer"]:
        lines.append(f"{row['dataset']} / {row['layer']} / P_{row['fixed_condition']}: "
                     + ", ".join(f"{key}={row[key]} (n={row[key + '_valid_tasks']})" for key in MEASURES))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--data_seed", type=int, required=True,
                        help="Training run's data seed; old checkpoints do not store it")
    parser.add_argument("--sampling_seed", type=int, default=12345)
    parser.add_argument("--num_tasks", type=int, default=100, help="Total task attempts, not per dataset")
    parser.add_argument("--num_repeats", type=int, default=3, help="Support pairs within each task")
    parser.add_argument("--output_dir", required=True, help="Must be a new directory")
    parser.add_argument("--image_size", type=int, default=128, help="Match the original training runner")
    parser.add_argument("--min_norm_sq", type=float, default=1e-20,
                        help="Squared-energy floor; no epsilon is added to any ratio")
    args = parser.parse_args(argv)
    if any(not 0 <= s < 2**32 for s in (args.data_seed, args.sampling_seed)):
        parser.error("Seeds must be uint32")
    if min(args.num_tasks, args.num_repeats, args.image_size) < 1:
        parser.error("Task/repeat counts and image size must be positive")
    if not math.isfinite(args.min_norm_sq) or args.min_norm_sq <= 0:
        parser.error("--min_norm_sq must be finite and positive")
    output = Path(args.output_dir).resolve()
    if output.exists():
        parser.error("--output_dir already exists; choose a new directory (no overwrite)")
    learner = MyLearner()
    learner.load(args.checkpoint)
    validate_checkpoint_config(learner.config)
    seed_evidence = {}
    for label, section in (("config", learner.config),
                           ("experiment_config", learner.config.get("experiment_config", {}))):
        if "data_seed" in section:
            seed_evidence[label + ".data_seed"] = section["data_seed"]
            if type(section["data_seed"]) is not int or section["data_seed"] != args.data_seed:
                raise ValueError(f"--data_seed differs from saved {label}.data_seed")
    if learner.transport.names != [name for name, _ in learner.learner.named_parameters()]:
        raise ValueError("Encoder and transport parameter orders differ")
    config_before = copy.deepcopy(learner.config)
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "max-va.pth"
    # The SECOND return value is meta-validation, using the training split seed.
    _, valid_info, _ = prepare_datasets_information(
        args.input_data_dir, learner.config["validation_datasets"], args.data_seed, False)
    datasets = create_datasets(valid_info, args.image_size)
    if not datasets:
        raise ValueError("No meta-validation datasets")
    inventories = [class_inventory(dataset) for dataset in datasets]
    pools, inventory = [x[0] for x in inventories], [x[1] for x in inventories]
    for report in inventory:
        print(f"{report['dataset']}: {report['eligible_classes']}/{report['total_classes']} "
              "classes have at least 30 usable images", flush=True)
    manifest = make_manifest(datasets, pools, args.num_tasks, args.num_repeats, args.sampling_seed)
    metadata = dict(status="running", checkpoint=str(checkpoint), checkpoint_sha256=digest_file(checkpoint),
        checkpoint_config=config_before, architecture=learner.transport.architecture(),
        best_validation_accuracy=learner.best_score, cli=vars(args), device=str(learner.dev),
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        data_seed_source=("checked against saved config data_seed" if seed_evidence else
                          "required CLI assertion from training log; not stored in checkpoint"),
        saved_data_seed_evidence=seed_evidence,
        split="meta-validation", index_base=0, index_definition="zero-based labels.csv row / ImageDataset index",
        split_file_sha256=digest_file(Path(args.input_data_dir) / "info/meta_splits.txt"),
        validation_datasets={name: dict(info=list(info), labels_sha256=digest_file(info[3]))
                             for name, info in valid_info.items()},
        protocol=dict(ways=WAYS, shots=SHOTS, query_per_class=QUERIES, inner_steps=STEPS,
            task_count="total attempts; uniform dataset selection; uniform eligible class selection",
            classes_and_query_fixed_across_repeats=True, within_pair_A_B_query_disjoint=True,
            across_pair_support_reuse_allowed=True, minimum_unique_images_per_class=30,
            duplicate_path_policy="exclude all rows whose resolved file path occurs more than once",
            missing_file_policy="exclude missing file paths before class eligibility; report counts",
            skipped_tasks_replaced=False, official_benchmark=False),
        numerical_policy=dict(norm_accumulation="float64", transport_arithmetic="checkpoint dtype",
            min_norm_sq=args.min_norm_sq, invalid_csv="empty value plus status", invalid_json=None,
            amplification_requires="valid r0/rP and raw disagreement squared energy above floor",
            epsilon_added=False),
        interpretation=["Support differences include prototype-head and functional BN batch-statistics changes.",
            "P includes learned scalar gate AND low-rank residual. P_A/P_B each stay fixed for mean and difference.",
            "r_system uses P_A(g_A) versus P_B(g_B); it is separate from fixed-P amplification.",
            "LR-off zeros the whole low-rank residual; learned scalar sigmoid(logit) remains active.",
            "Accuracy gain = on - off; loss gain = off - on; CE uses logits and natural logarithms.",
            "Task-level arithmetic means of valid pair ratios; dataset means weight tasks equally.",
            "Partial-valid repeats use available ratios; valid counts are reported. No epsilon imputation.",
            "Correlations use task means within each dataset, not independent support repeats; no causality."],
        data_source_sha256={name: digest_file(ROOT / name) for name in (
            "cdmetadl/helpers/general_helpers.py", "cdmetadl/ingestion/image_dataset.py",
            "cdmetadl/ingestion/data_generator.py")},
        source_sha256={name: digest_file(Path(__file__).with_name(name)) for name in (
            Path(__file__).name, "helpers_fo_proto_tclrsgmaml.py", "low_rank_transport.py",
            "task_transport.py", "model.py", "network.py", "eval_inner_steps.py")})
    task_rows, layer_means = [], []
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        files = {name: stack.enter_context((output / name).open("x", encoding="utf-8", newline=""))
                 for name in ("layer_pairs.csv", "support_benefits.csv", "task_benefits.csv",
                              "task_layer_means.csv", "sampling_manifest.json", "metadata.json",
                              "summary.json", "summary.txt")}
        layer_writer = write_csv(files["layer_pairs.csv"], LAYER_FIELDS, [])
        score_writer = write_csv(files["support_benefits.csv"], PAIR_FIELDS, [])
        try:
            # State equality is checked even on exceptions, with RNG restored.
            with observational_diagnostics(learner):
                for task in manifest:
                    if task["status"] == "skipped":
                        continue
                    task["status"] = "running"
                    try:
                        rows, scores, task_row = evaluate_task(
                            learner, datasets[task["dataset_index"]], task, args.min_norm_sq)
                    except ImageLoadError as exc:
                        task.update(status="skipped", skip_reason=str(exc))
                        continue
                    layer_writer.writerows(rows)
                    score_writer.writerows(scores)
                    files["layer_pairs.csv"].flush()
                    files["support_benefits.csv"].flush()
                    layer_means.extend(task_layer_means(rows))
                    task_rows.append(task_row)
                    task["status"] = "completed"
                    print(f"Task {task['task_id'] + 1}/{args.num_tasks}: {task['dataset']}; "
                          f"accuracy gain={task_row['accuracy_gain']:+.6f}, "
                          f"loss gain={task_row['loss_gain']:+.6f}", flush=True)
                if learner.config != config_before:
                    raise RuntimeError("Diagnostic changed checkpoint config")
                if any(p.grad is not None for root in (learner.learner, learner.transport) for p in root.parameters()):
                    raise RuntimeError("Diagnostic accumulated persistent parameter gradients")
            metadata.update(status="completed" if task_rows else "no_eligible_completed_tasks",
                            state_equality_checked=True, no_outer_gradients_checked=True)
        except BaseException as exc:
            metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            for task in manifest:
                if task["status"] == "running":
                    task["status"] = "failed"
            metadata["task_status_counts"] = dict(Counter(t["status"] for t in manifest))
            summary = summarize(task_rows, layer_means, manifest, inventory)
            summary["run_status"] = metadata["status"]
            write_csv(files["task_benefits.csv"], TASK_FIELDS, task_rows)
            fields = ("task_id", "dataset", "layer", "fixed_condition", "repeats") + MEASURES + tuple(
                key + "_valid_pairs" for key in MEASURES)
            write_csv(files["task_layer_means.csv"], fields, layer_means)
            for filename, value in (("metadata.json", metadata), ("summary.json", summary),
                                    ("sampling_manifest.json", dict(class_inventory=inventory, tasks=manifest))):
                json.dump(value, files[filename], indent=2, allow_nan=False)
                files[filename].write("\n")
            files["summary.txt"].write(summary_text(summary))
    print(summary_text(summary))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
