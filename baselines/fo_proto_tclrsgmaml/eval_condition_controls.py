r"""Paired, evaluation-only coefficient controls for an LR-only TC checkpoint.

Example (from the repository root)::

    python baselines/fo_proto_tclrsgmaml/eval_condition_controls.py \
        --checkpoint /path/to/max-va.pth \
        --input_data_dir /content/meta_album_feedback --seed 94 \
        --test_tasks_per_dataset 100 --shuffle_seed 12345 \
        --num_permutations 20 --output_dir /path/to/condition_controls

Uses the unchanged baseline adapt() through a local transport view. Episodes
are regenerated with the runner's protocol and checked against SHA-256 hashes
of the complete support/query tensors, labels and class metadata on EVERY pass.
Only fingerprints and coefficients are cached, not all episode images.

Accuracies/differences in files are fractions; the terminal uses percent/pp.
Task/permutation indices are zero-based. Overall accuracy weights tasks equally.
The mean control uses all evaluation SUPPORT-derived vectors (a transductive
analysis control, not a deployable independently fitted baseline).
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers_fo_proto_tclrsgmaml import adapt  # noqa: E402


TEST_EPISODES = dict(N=None, min_N=2, max_N=20, k=None, min_k=1, max_k=20,
                     query_images_per_class=20)
DIAGNOSTICS = ("correction_to_gradient_ratio", "transport_to_gradient_ratio",
               "cos_gradient_correction", "cos_gradient_transport")
INTERPRETATION = [
    "Shuffle degradation alone does not establish a benefit from task conditioning: "
    "large corrections can make mismatched coefficients harmful.",
    "OWN > MEAN supports a benefit of task-specific variation on these episodes; "
    "consider the paired uncertainty and effect size.",
    "OWN > WITHIN SHUFFLE supports the importance of the correct pairing within "
    "a dataset, but sensitivity to incorrect coefficients remains an alternative.",
    "GLOBAL SHUFFLE < WITHIN SHUFFLE is consistent with domain-level information; "
    "global shuffle degradation by itself must not be labelled domain information.",
    "OWN approximately equal to MEAN with poor SHUFFLE can reflect sensitivity to "
    "wrong coefficients rather than a benefit from task-specific variation.",
    "MEAN preserves the mean residual, not the distribution of coefficient norms. "
    "The shuffles preserve the coefficient multiset and magnitude distribution exactly.",
]


def require(condition, message):
    """Keep scientific validity checks active even under python -O."""
    if not condition:
        raise ValueError(message)


class RankCodec:
    """Flatten in checkpoint rank_layout order, never sorted ParameterDict order."""

    def __init__(self, architecture):
        tc = architecture["task_conditioning"]
        self.rows = copy.deepcopy(tc["rank_layout"])
        self.scalar_names = list(tc["scalar_names"])
        offset = len(self.scalar_names)
        keys = set()
        for row in self.rows:
            key, rank = row["key"], row["rank"]
            require(key not in keys and rank > 0, "Invalid/duplicate rank layout entry")
            require(row["start"] == offset and row["stop"] == offset + rank,
                    "Checkpoint rank layout is not contiguous")
            require(0 <= int(key) < len(self.scalar_names)
                    and self.scalar_names[int(key)] == row["name"],
                    "Rank layout name/key mismatch")
            offset += rank
            keys.add(key)
        require(self.rows and offset == tc["output_size"], "Invalid rank layout size")
        self.size = offset - len(self.scalar_names)
        self.keys = keys

    def flatten(self, delta_c):
        require(set(delta_c) == self.keys, "Coefficient layer keys do not match checkpoint")
        parts = []
        for row in self.rows:
            value = delta_c[row["key"]].detach().cpu()
            require(tuple(value.shape) == (row["rank"],), "Coefficient rank mismatch")
            parts.append(value)
        require(len({x.dtype for x in parts}) == 1, "Mixed coefficient dtypes")
        vector = torch.cat(parts).clone()
        require(bool(torch.isfinite(vector).all()), "Nonfinite coefficient")
        return vector

    def reconstruct(self, vector, like):
        vector = torch.as_tensor(vector, device=like.device, dtype=like.dtype)
        require(tuple(vector.shape) == (self.size,), "Flattened coefficient size mismatch")
        require(bool(torch.isfinite(vector).all()), "Nonfinite coefficient override")
        base = len(self.scalar_names)
        return {r["key"]: vector[r["start"] - base:r["stop"] - base].detach().clone()
                for r in self.rows}

    def check_round_trip(self, vector, delta_c):
        rebuilt = self.reconstruct(vector, next(iter(delta_c.values())))
        for key in self.keys:
            require(torch.equal(rebuilt[key], delta_c[key].detach()),
                    f"Flatten/reconstruct changed layer {key}")
        require(torch.equal(self.flatten(rebuilt), vector), "Coefficient round-trip failed")


class CorrectionDiagnostics:
    """Whole-encoder norms/cosines per step, averaged across inner steps.

    G is the clipped encoder support gradient. C is recovered observationally
    as G_tilde - sigmoid(a) * G in the model dtype (zero on unranked tensors).
    Moments are accumulated in float64. No diagnostic enters an update.
    Zero denominators produce null, with valid step counts reported separately.
    """

    def __init__(self, transport):
        self.transport = transport
        self.moments = None
        self.steps = []
        self.position = 0

    @torch.no_grad()
    def record(self, name, gradient, transformed):
        require(name == self.transport.names[self.position], "Encoder gradient order changed")
        key = self.transport.indices[name]
        g, t = gradient.detach(), transformed.detach()
        c = (t - self.transport.logits[key].sigmoid() * g
             if key in self.transport.u else torch.zeros_like(g))
        g, c, t = g.double(), c.double(), t.double()
        values = torch.stack((g.square().sum(), c.square().sum(), t.square().sum(),
                              (g * c).sum(), (g * t).sum()))
        self.moments = values if self.moments is None else self.moments + values
        self.position += 1
        if self.position == len(self.transport.names):
            self.steps.append(self.moments)
            self.moments = None
            self.position = 0

    def result(self):
        require(self.position == 0, "Incomplete encoder diagnostic step")
        values = {key: [] for key in DIAGNOSTICS}
        if self.steps:
            for g2, c2, t2, gc, gt in torch.stack(self.steps).cpu().numpy():
                require(np.isfinite([g2, c2, t2, gc, gt]).all(), "Nonfinite gradient diagnostics")
                g, c, t = np.sqrt([g2, c2, t2])
                if g > 0:
                    values[DIAGNOSTICS[0]].append(float(c / g))
                    values[DIAGNOSTICS[1]].append(float(t / g))
                if g * c > 0:
                    values[DIAGNOSTICS[2]].append(float(np.clip(gc / (g * c), -1, 1)))
                if g * t > 0:
                    values[DIAGNOSTICS[3]].append(float(np.clip(gt / (g * t), -1, 1)))
        result = {key: float(np.mean(v)) if v else None for key, v in values.items()}
        result.update({key + "_valid_steps": len(v) for key, v in values.items()})
        result["inner_steps"] = len(self.steps)
        return result


class EvaluationTransport:
    """Local duck-typed transport view consumed by the unchanged adapt helper."""

    def __init__(self, transport, codec, conditioning_override=None, diagnostics=True):
        require(not transport.training, "Transport must be in evaluation mode")
        self.transport, self.codec = transport, codec
        self.names = transport.names
        self.override = conditioning_override
        self.vector = None
        self.condition_calls = 0
        self.diagnostics = CorrectionDiagnostics(transport) if diagnostics else None

    def condition(self, support_embedding):
        self.condition_calls += 1
        require(self.condition_calls == 1, "Conditioning must be computed once per task")
        if self.override is None:
            conditioning = self.transport.condition(support_embedding)
            require(conditioning is not None, "OWN requires enabled GateNet conditioning")
            delta_a, delta_c = conditioning
            require(bool((delta_a == 0).all()), "LR-only TC requires delta_a=0")
        else:
            # Do not call GateNet in MEAN/SHUFFLE. Prototype support features are
            # still computed by adapt exactly as in the normal baseline.
            like = next(self.transport.logits.parameters())
            delta_a = like.new_zeros(len(self.names))
            delta_c = self.codec.reconstruct(self.override, like)
        self.vector = self.codec.flatten(delta_c)
        self.codec.check_round_trip(self.vector, delta_c)
        if self.override is not None:
            require(torch.equal(self.vector, torch.as_tensor(self.override).detach().cpu()),
                    "Applied override differs from supplied coefficient")
        return delta_a, delta_c

    def transport_gradient(self, name, gradient, conditioning):
        transformed = self.transport.transport_gradient(name, gradient, conditioning)
        if self.diagnostics is not None:
            self.diagnostics.record(name, gradient, transformed)
        return transformed


@torch.no_grad()
def evaluate_task(learner, task, codec, conditioning_override=None, diagnostics=True):
    """Query tensors are read only AFTER support-only adaptation has completed."""
    view = EvaluationTransport(learner.transport, codec, conditioning_override, diagnostics)
    support, labels, _ = task.support_set
    fast = adapt(learner.learner, list(learner.learner.parameters()),
                 support.to(learner.dev), labels.to(learner.dev),
                 learner.config["method_config"], task.num_ways, view)
    require(view.condition_calls == 1, "Missing task conditioning")
    probabilities = learner.learner.forward_weights(
        task.query_set[0].to(learner.dev), fast).softmax(1).cpu().numpy()
    require(np.isfinite(probabilities).all(), "Nonfinite query predictions")
    accuracy = float(np.mean(probabilities.argmax(1) == task.query_set[1].cpu().numpy()))
    diagnostic_values = view.diagnostics.result() if diagnostics else {}
    if diagnostics:
        require(diagnostic_values["inner_steps"] == learner.config["method_config"]["inner_steps"],
                "Diagnostic count does not match checkpoint inner steps")
    return accuracy, view.vector, diagnostic_values, probabilities


def validate_lr_only(learner):
    config = learner.config
    tc, method = config["task_conditioning"], config["method_config"]
    require(tc["enabled"] is True and config["lrsg"]["enabled"] is True,
            "An enabled LR-only TC checkpoint is required")
    require(tc["scalar_delta_scale"] == 0.0 and tc["low_rank_delta_scale"] == 1.0,
            "Checkpoint must have scalar_delta_scale=0 and low_rank_delta_scale=1; "
            "the analysis never edits checkpoint config")
    require(method["first_order"] is True and method["reset_classifier"] is False,
            "Expected first-order prototype adaptation")
    require(learner.transport.scalar_scale == 0.0 and learner.transport.low_rank_scale == 1.0,
            "Loaded transport is not LR-only TC")
    require(learner.transport.architecture() == learner.state["architecture"],
            "Checkpoint architecture mismatch")
    require(not learner.learner.training and not learner.transport.training,
            "Learner and transport must be in evaluation mode")


class StateGuard:
    """Exact state comparisons include encoder buffers, U/V, logits and GateNet."""

    def __init__(self, learner):
        self.learner = learner
        self.config = copy.deepcopy(learner.config)
        self.states = {
            name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in self.modules()
        }
        self.checks = 0

    def modules(self):
        return (("encoder", self.learner.learner), ("transport", self.learner.transport))

    def check(self):
        require(self.learner.config == self.config, "Checkpoint adaptation config changed")
        for name, module in self.modules():
            current = module.state_dict()
            require(current.keys() == self.states[name].keys(), f"{name} state keys changed")
            for key, original in self.states[name].items():
                require(torch.equal(current[key].detach().cpu(), original), f"State changed: {name}.{key}")
            require(all(p.grad is None for p in module.parameters()),
                    f"Unexpected accumulated outer gradients in {name}")
        self.checks += 1


@contextmanager
def replay_rng(seed):
    """Reset all episode RNGs per pass and restore caller RNGs on exit/failure."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def array_digest(value):
    digest = hashlib.sha256()
    if value is None:
        return None
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    array = np.ascontiguousarray(array)
    digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def task_identity(task, index):
    return dict(task_index=index, dataset=str(task.dataset), ways=int(task.num_ways),
                shots=int(task.num_shots),
                support_sha256=[array_digest(x) for x in task.support_set],
                query_sha256=[array_digest(x) for x in task.query_set],
                original_classes_sha256=array_digest(getattr(task, "original_class_idx", None)))


class CheckedEpisodes:
    def __init__(self, factory, seed, expected_count=None):
        self.factory, self.seed, self.expected_count = factory, seed, expected_count
        self.manifest = []
        self.completed_passes = 0

    def iterate(self, record=False):
        require((record and not self.manifest and not self.completed_passes)
                or (not record and self.completed_passes > 0), "Invalid episode replay order")
        count = 0
        with replay_rng(self.seed):
            for i, task in enumerate(self.factory()):
                identity = task_identity(task, i)
                if record:
                    self.manifest.append(identity)
                else:
                    require(i < len(self.manifest) and identity == self.manifest[i],
                            f"Episode identity mismatch at task {i}")
                count += 1
                yield i, task
        require(count > 0, "No test episodes")
        if self.expected_count is not None:
            require(count == self.expected_count, "Unexpected number of test episodes")
        require(count == len(self.manifest), "Episode replay ended early")
        self.completed_passes += 1


def derangement(size, rng):
    """Rejection sampling is uniform over ALL exact derangements (not just cycles)."""
    require(size >= 2, "Exact derangement requires at least two tasks per group")
    identity = np.arange(size)
    while True:
        permutation = rng.permutation(size)
        if np.all(permutation != identity):
            return permutation


def derangement_count(size, cap):
    """Count only up to the requested ensemble size; avoid enormous factorials."""
    previous, current = 1, 0  # !0, !1
    if size == 0:
        return 1
    for n in range(2, size + 1):
        previous, current = current, (n - 1) * (previous + current)
        if current >= cap:
            return cap
    return current


def check_permutation(permutation, datasets, coefficients=None, within=False):
    n = len(datasets)
    require(permutation.shape == (n,) and np.issubdtype(permutation.dtype, np.integer),
            "Invalid permutation shape/dtype")
    require(np.array_equal(np.sort(permutation), np.arange(n)), "Permutation is not a bijection")
    require(np.all(permutation != np.arange(n)), "Permutation contains a self-assignment")
    datasets = np.asarray(datasets)
    if within:
        require(np.array_equal(datasets[permutation], datasets), "Within shuffle crosses datasets")
    if coefficients is not None:
        # Invert the assignment: exact source order must be recoverable, including duplicates.
        require(np.array_equal(coefficients[permutation][np.argsort(permutation)], coefficients),
                "Coefficient multiset changed")
        if within:
            for dataset in dict.fromkeys(datasets.tolist()):
                members = np.flatnonzero(datasets == dataset)
                require(np.array_equal(np.sort(permutation[members]), members),
                        "Within-dataset multiset changed")


def make_permutations(datasets, number, shuffle_seed, coefficients=None):
    require(number >= 1 and shuffle_seed >= 0, "Positive permutation count and nonnegative seed required")
    datasets = np.asarray(datasets)
    groups = [np.flatnonzero(datasets == name) for name in dict.fromkeys(datasets.tolist())]
    require(groups and all(len(group) >= 2 for group in groups),
            "Within-dataset derangement is impossible for a singleton dataset; use at least two tasks per dataset")
    available_within = 1
    for group in groups:
        available_within = min(number, available_within * derangement_count(len(group), number))
    require(derangement_count(len(datasets), number) >= number and available_within >= number,
            "Too few distinct derangements for --num_permutations; reduce that value "
            "or increase the number of tasks per dataset")
    global_rows, within_rows = [], []
    seen_global, seen_within = set(), set()
    for pid in range(number):
        # Separate streams preserve the prefix when --num_permutations increases.
        global_rng = np.random.default_rng(np.random.SeedSequence([shuffle_seed, 0, pid]))
        within_rng = np.random.default_rng(np.random.SeedSequence([shuffle_seed, 1, pid]))
        while True:
            global_row = derangement(len(datasets), global_rng)
            if global_row.tobytes() not in seen_global:
                seen_global.add(global_row.tobytes())
                break
        while True:
            within_row = np.empty(len(datasets), dtype=np.int64)
            for group in groups:
                within_row[group] = group[derangement(len(group), within_rng)]
            if within_row.tobytes() not in seen_within:
                seen_within.add(within_row.tobytes())
                break
        check_permutation(global_row, datasets, coefficients)
        check_permutation(within_row, datasets, coefficients, within=True)
        global_rows.append(global_row)
        within_rows.append(within_row)
    return np.stack(global_rows), np.stack(within_rows)


def paired_statistics(own, control):
    from scipy.stats import t

    differences = np.asarray(own, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    require(differences.ndim == 1 and differences.size > 0
            and np.isfinite(differences).all(), "Invalid paired differences")
    n = len(differences)
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    half = float(t.ppf(.975, n - 1) * se) if se is not None else None
    return dict(n=n, mean_paired_difference=mean, standard_error=se,
                ci95=[mean - half, mean + half] if half is not None else None,
                ci_method="Student t interval over paired task differences (df=N-1)",
                fraction_own_greater=float(np.mean(differences > 0)),
                fraction_own_less=float(np.mean(differences < 0)),
                fraction_equal=float(np.mean(differences == 0)))


def shuffle_statistics(own, shuffled):
    own = np.asarray(own, dtype=np.float64)
    shuffled = np.asarray(shuffled, dtype=np.float64)
    require(shuffled.ndim == 2 and shuffled.shape[1] == len(own)
            and len(shuffled) > 0 and np.isfinite(shuffled).all(), "Invalid shuffle accuracies")
    null = shuffled.mean(axis=1)
    count = int(np.count_nonzero(null >= own.mean()))
    return dict(mean_accuracy=float(null.mean()), std_accuracy=float(null.std(ddof=0)),
                std_ddof=0, min_accuracy=float(null.min()), max_accuracy=float(null.max()),
                null_distribution=null.tolist(), num_permutations=len(null),
                num_shuffle_accuracies_ge_own=count,
                permutation_p_value=(1 + count) / (1 + len(null)),
                minimum_attainable_p=1 / (1 + len(null)),
                paired_by_permutation=[dict(permutation_id=i, **paired_statistics(own, row))
                                       for i, row in enumerate(shuffled)])


def accuracy_summary(own, mean, global_acc, within_acc):
    own_mean, mean_mean = float(np.mean(own)), float(np.mean(mean))
    global_stats, within_stats = shuffle_statistics(own, global_acc), shuffle_statistics(own, within_acc)
    return dict(own=own_mean, mean=mean_mean,
                global_shuffle_mean=global_stats["mean_accuracy"],
                within_shuffle_mean=within_stats["mean_accuracy"],
                own_minus_mean=own_mean - mean_mean,
                own_minus_global=own_mean - global_stats["mean_accuracy"],
                own_minus_within=own_mean - within_stats["mean_accuracy"],
                mean_minus_global=mean_mean - global_stats["mean_accuracy"],
                mean_minus_within=mean_mean - within_stats["mean_accuracy"],
                paired_own_vs_mean=paired_statistics(own, mean),
                shuffle_global=global_stats, shuffle_within=within_stats)


def aggregate_diagnostics(rows):
    result = {}
    for key in DIAGNOSTICS:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[key + "_valid_tasks"] = len(values)
    return result


def diagnostic_summary(rows, datasets):
    return dict(overall=aggregate_diagnostics(rows), by_dataset={
        name: aggregate_diagnostics([row for row, label in zip(rows, datasets) if label == name])
        for name in dict.fromkeys(datasets)})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def progress(label, done, total):
    if done == 1 or done % 25 == 0 or done == total:
        print(f"{label}: {done}/{total if total is not None else '?'} tasks", flush=True)


def run_analysis(learner, episodes, output_dir, num_permutations=20, shuffle_seed=12345,
                 metadata=None):
    """Run OWN (plus normal-path parity), MEAN and both permutation ensembles."""
    validate_lr_only(learner)
    require(num_permutations >= 1 and shuffle_seed >= 0, "Invalid permutation settings")
    codec = RankCodec(learner.state["architecture"])
    guard = StateGuard(learner)
    guard.check()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    own, vectors, own_diagnostics = [], [], []
    for i, task in episodes.iterate(record=True):
        acc, vector, diag, probabilities = evaluate_task(learner, task, codec)
        # Check EVERY OWN task through the public normal evaluation path too.
        reference = learner.fit((*task.support_set, task.num_ways, task.num_shots)).predict(task.query_set[0])
        require(np.array_equal(probabilities, reference), f"OWN/normal evaluation mismatch at task {i}")
        own.append(acc)
        vectors.append(vector)
        own_diagnostics.append(diag)
        progress("OWN + normal-path parity", i + 1, episodes.expected_count)
    guard.check()
    n = len(episodes.manifest)
    require(len(vectors) == n, "OWN coefficient count differs from task count")
    coefficients = torch.stack(vectors)
    mean_vector = coefficients.mean(dim=0)
    coefficients_np = coefficients.numpy()
    own = np.asarray(own, dtype=np.float64)
    datasets = [row["dataset"] for row in episodes.manifest]
    global_perms, within_perms = make_permutations(datasets, num_permutations, shuffle_seed, coefficients_np)
    np.savez_compressed(output_dir / "condition_vectors.npz",
                        own_delta_c=coefficients_np, mean_delta_c=mean_vector.numpy(),
                        dataset_labels=np.asarray(datasets), task_index=np.arange(n),
                        global_permutation_indices=global_perms, within_permutation_indices=within_perms,
                        rank_layout_json=np.asarray(json.dumps(codec.rows)),
                        architecture_json=np.asarray(json.dumps(learner.state["architecture"])),
                        seed=np.asarray(episodes.seed), shuffle_seed=np.asarray(shuffle_seed))
    write_json(output_dir / "condition_task_manifest.json", episodes.manifest)

    mean, mean_diagnostics = [], []
    main_fields = ["task_index", "dataset", "ways", "shots", "own_accuracy", "mean_accuracy"]
    main_fields += [f"{mode}_{key}" for mode in ("own", "mean") for key in DIAGNOSTICS]
    with (output_dir / "condition_control_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_fields)
        writer.writeheader()
        for i, task in episodes.iterate():
            acc, vector, diag, _ = evaluate_task(learner, task, codec, mean_vector)
            require(torch.equal(vector, mean_vector), f"MEAN override differs at task {i}")
            mean.append(acc)
            mean_diagnostics.append(diag)
            row = {key: episodes.manifest[i][key] for key in main_fields[:4]}
            row.update(own_accuracy=own[i], mean_accuracy=acc)
            for mode, values in (("own", own_diagnostics[i]), ("mean", diag)):
                row.update({f"{mode}_{key}": values[key] for key in DIAGNOSTICS})
            writer.writerow(row)
            progress("MEAN", i + 1, n)
    guard.check()
    mean = np.asarray(mean, dtype=np.float64)

    shuffle_accuracies, shuffle_diagnostics = {}, {}
    shuffle_fields = ["mode", "permutation_id", "task_index", "dataset", "ways", "shots",
                      "accuracy", "own_accuracy", "difference_from_own", "assigned_from_task_index"]
    shuffle_fields += list(DIAGNOSTICS)
    with (output_dir / "condition_shuffle_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=shuffle_fields)
        writer.writeheader()
        for mode, permutations in (("shuffle_global", global_perms), ("shuffle_within", within_perms)):
            accuracies = np.empty((num_permutations, n), dtype=np.float64)
            diagnostic_rows = []
            for pid, permutation in enumerate(permutations):
                current_diagnostics = []
                check_permutation(permutation, datasets, coefficients_np, within=mode == "shuffle_within")
                for i, task in episodes.iterate():
                    source = int(permutation[i])
                    acc, vector, diag, _ = evaluate_task(learner, task, codec, coefficients[source])
                    require(torch.equal(vector, coefficients[source]), "Shuffle override mismatch")
                    accuracies[pid, i] = acc
                    current_diagnostics.append(diag)
                    row = {key: episodes.manifest[i][key] for key in main_fields[:4]}
                    row.update(mode=mode, permutation_id=pid, accuracy=acc, own_accuracy=own[i],
                               difference_from_own=acc - own[i], assigned_from_task_index=source)
                    row.update({key: diag[key] for key in DIAGNOSTICS})
                    writer.writerow(row)
                    progress(f"{mode} {pid + 1}/{num_permutations}", i + 1, n)
                guard.check()
                handle.flush()
                diagnostic_rows.extend(current_diagnostics)
            shuffle_accuracies[mode] = accuracies
            shuffle_diagnostics[mode] = diagnostic_summary(diagnostic_rows, datasets * num_permutations)

    global_acc, within_acc = shuffle_accuracies["shuffle_global"], shuffle_accuracies["shuffle_within"]
    summary = dict(
        metadata=dict(metadata or {}, seed=episodes.seed, shuffle_seed=shuffle_seed,
                      num_tasks=n, num_permutations=num_permutations,
                      tasks_per_dataset=dict(Counter(datasets)), task_index_base=0,
                      accuracy_units="fraction", accuracy_weighting="equal weight per task",
                      difference_from_own_sign="control minus own",
                      method_config=copy.deepcopy(learner.config["method_config"]),
                      architecture=learner.state["architecture"],
                      same_task_method="fresh seeded generator + full support/query SHA-256 equality"),
        overall=accuracy_summary(own, mean, global_acc, within_acc), by_dataset={},
        diagnostics=dict(own=diagnostic_summary(own_diagnostics, datasets),
                         mean=diagnostic_summary(mean_diagnostics, datasets), **shuffle_diagnostics),
        diagnostics_definition=CorrectionDiagnostics.__doc__,
        diagnostics_aggregation="mean over inner steps, then equal weight per task and permutation; "
                                "undefined zero-denominator values are excluded and counts reported",
        permutation_note=f"Exploratory permutation-style comparison over distinct derangements "
                         f"sampled uniformly without replacement. With "
                         f"{num_permutations} permutations minimum attainable p and resolution are "
                         f"1/{1 + num_permutations} = {1 / (1 + num_permutations):.6f}. "
                         "Use 99 or 199 permutations for finer resolution. These Monte Carlo "
                         "values are not a guarantee of an exact randomization test: derangements "
                         "exclude identity/fixed points and require exchangeability assumptions.",
        uncertainty_note="Task-level Student t CIs are descriptive, unadjusted for multiple "
                         "comparisons, dataset clustering or shared MEAN estimation. Do not pool "
                         "permutations as independent tasks.",
        interpretation_notes=INTERPRETATION,
    )
    for name in dict.fromkeys(datasets):
        mask = np.asarray(datasets) == name
        summary["by_dataset"][name] = dict(num_tasks=int(mask.sum()), **accuracy_summary(
            own[mask], mean[mask], global_acc[:, mask], within_acc[:, mask]))
    guard.check()
    summary["sanity_checks"] = dict(
        own_matches_normal_predictions_every_task=True, own_coefficient_count=n,
        flatten_reconstruct_exact=True, global_and_within_multisets_preserved=True,
        distinct_permutations_per_shuffle_mode=True,
        no_self_assignments=True, within_dataset_membership_preserved=True,
        same_mean_vector_every_task=True, verified_episode_passes=episodes.completed_passes,
        expected_episode_passes=2 + 2 * num_permutations,
        encoder_buffers_and_parameters_unchanged=True, uv_unchanged=True,
        scalar_logits_unchanged=True, gate_net_unchanged=True, state_comparisons=guard.checks,
        no_outer_gradients_accumulated=True, no_optimizer_created_or_stepped=True,
        support_only_conditioning=True, override_bypasses_gate_net=True,
        unchanged_baseline_adapt=True, create_graph=False, classifier_transport=False,
        checkpoint_inner_steps_lrs_and_clipping_unchanged=True)
    require(episodes.completed_passes == 2 + 2 * num_permutations, "Incomplete evaluation passes")
    dataset_fields = ["dataset", "num_tasks", "own", "mean", "global_shuffle_mean", "within_shuffle_mean",
                      "own_minus_mean", "own_minus_global", "own_minus_within"]
    with (output_dir / "condition_dataset_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset_fields)
        writer.writeheader()
        for name, values in summary["by_dataset"].items():
            writer.writerow(dict(dataset=name, **{key: values[key] for key in dataset_fields[1:]}))
    write_json(output_dir / "condition_control_summary.json", summary)
    print_summary(summary)
    return summary


def print_summary(summary):
    overall = summary["overall"]
    print("\nAccuracy (%) and paired differences (percentage points):")
    print(f"OWN {100 * overall['own']:.4f}   MEAN {100 * overall['mean']:.4f}")
    for mode in ("shuffle_global", "shuffle_within"):
        stats = overall[mode]
        print(f"{mode}: mean={100 * stats['mean_accuracy']:.4f}, "
              f"std={100 * stats['std_accuracy']:.4f}, "
              f"min/max={100 * stats['min_accuracy']:.4f}/{100 * stats['max_accuracy']:.4f}, "
              f"p={stats['permutation_p_value']:.6f}")
    for key in ("own_minus_mean", "own_minus_global", "own_minus_within", "mean_minus_global", "mean_minus_within"):
        print(f"{key}: {100 * overall[key]:+.4f} pp")
    paired = overall["paired_own_vs_mean"]
    ci = paired["ci95"]
    print(f"OWN - MEAN paired: mean={100 * paired['mean_paired_difference']:+.4f} pp, "
          f"SE={None if paired['standard_error'] is None else 100 * paired['standard_error']}, "
          f"95% CI (pp)={None if ci is None else [100 * x for x in ci]}; "
          f"own > / < / = mean: {paired['fraction_own_greater']:.4f} / "
          f"{paired['fraction_own_less']:.4f} / {paired['fraction_equal']:.4f}")
    print("\ndataset                     OWN     MEAN   GLOBAL   WITHIN    O-M      O-G      O-W")
    keys = ("own", "mean", "global_shuffle_mean", "within_shuffle_mean", "own_minus_mean", "own_minus_global", "own_minus_within")
    for name, values in summary["by_dataset"].items():
        print(f"{name:<24}" + "".join(f"{100 * values[key]:9.3f}" for key in keys))
    print("\nWhole-encoder correction diagnostics (mean per step/task/permutation):")
    for mode, values in summary["diagnostics"].items():
        print(mode + ": " + ", ".join(f"{key}={values['overall'][key]}" for key in DIAGNOSTICS))
    print("\n" + summary["permutation_note"])
    print(summary["uncertainty_note"])
    for note in summary["interpretation_notes"]:
        print(note)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--test_tasks_per_dataset", type=int, default=100)
    parser.add_argument("--shuffle_seed", type=int, default=12345)
    parser.add_argument("--num_permutations", type=int, default=20)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.test_tasks_per_dataset < 2:
        parser.error("--test_tasks_per_dataset must be >= 2 for within-dataset derangement")
    if args.num_permutations < 1 or args.shuffle_seed < 0 or not 0 <= args.seed < 2**32:
        parser.error("Require positive --num_permutations, nonnegative --shuffle_seed and a uint32 --seed")
    if args.image_size < 1:
        parser.error("--image_size must be positive")

    from model import MyLearner
    from cdmetadl.helpers.general_helpers import prepare_datasets_information
    from cdmetadl.ingestion.image_dataset import create_datasets
    from cdmetadl.ingestion.data_generator import CompetitionDataLoader

    learner = MyLearner()
    learner.load(args.checkpoint)
    validate_lr_only(learner)
    _, _, test_info = prepare_datasets_information(
        args.input_data_dir, learner.config["validation_datasets"], args.seed, False, scoring=True)
    datasets = create_datasets(test_info, args.image_size)

    def factory():
        loader = CompetitionDataLoader(datasets, TEST_EPISODES, args.seed, test_generator=True)
        return loader.generator(args.test_tasks_per_dataset)

    episodes = CheckedEpisodes(factory, args.seed, len(datasets) * args.test_tasks_per_dataset)
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "max-va.pth"
    with checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    print(f"Loaded {checkpoint}; evaluating {episodes.expected_count} tasks on {learner.dev}, "
          f"seed={args.seed}, {args.num_permutations} permutations per shuffle mode.", flush=True)
    run_analysis(learner, episodes, args.output_dir, args.num_permutations, args.shuffle_seed,
                 metadata=dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
                               input_data_dir=str(Path(args.input_data_dir).resolve()),
                               image_size=args.image_size, test_episodes=TEST_EPISODES,
                               best_validation_accuracy=learner.best_score))
    print(f"\nAnalysis files written to {Path(args.output_dir).resolve()}", flush=True)


if __name__ == "__main__":
    main()
