r"""Direction x magnitude decomposition of the LR-only-TC encoder update.

At every inner step, ``g`` is the clipped support gradient of one encoder tensor
(the transport input) and ``Pg`` is the baseline's transported gradient::

    Pg = sigmoid(a) * g  +  gamma * beta * U diag(1 + delta_c) V^T g

Six encoder updates adapt the SAME task object from the SAME checkpoint weights
(the checkpoint itself is never modified)::

    plain        g                        direction g,  norm g
    mag_layer    g  * |Pg|_l / |g|_l      direction g,  norm of Pg, per tensor
    mag_global   g  * |Pg|   / |g|        direction g,  norm of Pg, one scalar
    dir_layer    Pg * |g|_l  / |Pg|_l     direction Pg, norm of g,  per tensor
    dir_global   Pg * |g|    / |Pg|       direction Pg, norm of g,  one scalar
    full         Pg                       the trained method

``|.|_l`` is the Frobenius norm of one parameter tensor, ``|.|`` the norm over
all encoder tensors; both are recomputed at every inner step from that variant's
own trajectory. Classifier-head updates are the ordinary FO-Proto-MAML steps in
all six variants. Reading the columns:

* ``acc_full`` must reproduce the scorer's ``task_results.csv`` produced at the
  same ``--gamma`` (reference parity, abs_tol=1e-12). The legacy final results
  were scored at gamma=1.
* ``acc_plain`` is the LRTC-trained encoder with raw gradient steps. It is NOT
  FO-Proto-MAML (different encoder) and differs from the gamma sweep's
  ``acc_gamma_0`` only by the learned scalar gates.
* ``full - mag_layer``: what the direction adds when every tensor already takes
  a step of the transported size. ``mag_layer`` gives the raw gradient a
  per-tensor, per-task, per-step step size taken from P itself.
* ``mag_global - plain``: isotropic step-size effect only.
* ``dir_layer - plain``: direction effect at the plain step size.
* ``acc_steps0`` is the prototype start of the same encoder (no adaptation).

The five counterfactual variants may diverge. A variant with nonfinite support
loss or nonfinite query predictions is scored with uniform probabilities (the
scorer's first-index tie rule, i.e. chance level on balanced queries) and
flagged in ``diverged_<variant>``. Nonfinite ``steps0``/``full`` predictions
are an error. ``support_loss_start``, ``norm_ratio_step1`` (|Pg| / |g|) and
``cos_g_pg_step1`` describe the shared first step.

Run from the repository root, using the original run's data, seed and image size::

    python baselines/fo_proto_tclrsgmaml/eval_direction_magnitude.py \
        --checkpoint /path/to/model/max-va.pth \
        --input_data_dir /content/meta_album_final --seed 93 \
        --image_size 128 --test_tasks_per_dataset 600 --gamma 1 \
        --reference_task_results /path/to/scoring/task_results.csv \
        --out_csv /path/to/lrtc_seed93_direction_magnitude.csv

Accuracies in the CSV are fractions; the printed summary uses percent and
percentage points and weights tasks equally. The variants are diagnostics of
one trained checkpoint on meta-test tasks, not separately trained methods.
"""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MyLearner, MyPredictor  # noqa: E402
from helpers_fo_proto_tclrsgmaml import adapt, prototype_head  # noqa: E402
from eval_inner_steps import (PARITY_ATOL, TEST_EPISODES, ReferenceTaskResults,  # noqa: E402
                              observational_diagnostics, validate_checkpoint_config)
from eval_beta_sweep import (ID_FIELDS, SHOT_BINS, new_output_csv, scorer_accuracy,  # noqa: E402
                             shot_bin)
from cdmetadl.helpers.general_helpers import prepare_datasets_information  # noqa: E402
from cdmetadl.ingestion.image_dataset import create_datasets  # noqa: E402
from cdmetadl.ingestion.data_generator import CompetitionDataLoader  # noqa: E402

VARIANTS = ("plain", "mag_layer", "mag_global", "dir_layer", "dir_global", "full")
COUNTERFACTUALS = VARIANTS[:-1]
STEP1_FIELDS = ("support_loss_start", "norm_ratio_step1", "cos_g_pg_step1")
ACCURACY_FIELDS = ("acc_steps0",) + tuple(f"acc_{v}" for v in VARIANTS)
DIVERGED_FIELDS = tuple(f"diverged_{v}" for v in COUNTERFACTUALS)
FIELDS = ID_FIELDS + ACCURACY_FIELDS + DIVERGED_FIELDS + STEP1_FIELDS
CONTRASTS = (("full", "plain"), ("mag_layer", "plain"), ("mag_global", "plain"),
             ("dir_layer", "plain"), ("dir_global", "plain"),
             ("full", "mag_layer"), ("full", "mag_global"))
SELF_CHECK_RTOL, SELF_CHECK_ATOL = 1e-5, 1e-8


def squared_norm(tensors):
    """Float64 squared Frobenius norm over a list of tensors."""
    return sum(t.detach().double().square().sum() for t in tensors)


def safe_ratio(numerator, denominator):
    """numerator / denominator, with x/0 -> 0 so that a zero gradient stays zero."""
    return torch.where(denominator > 0, numerator / denominator,
                       torch.zeros_like(numerator))


def variant_gradients(variant, raw, full):
    """Encoder update of one variant from the raw (g) and transported (Pg) lists."""
    if variant == "plain":
        return list(raw)
    if variant == "full":
        return list(full)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant!r}")
    kind, scope = variant.split("_")
    direction, magnitude = (raw, full) if kind == "mag" else (full, raw)
    if scope == "global":
        scale = safe_ratio(squared_norm(magnitude).sqrt(), squared_norm(direction).sqrt())
        return [d * scale.to(d.dtype) for d in direction]
    return [d * safe_ratio(squared_norm([m]).sqrt(), squared_norm([d]).sqrt()).to(d.dtype)
            for d, m in zip(direction, magnitude, strict=True)]


def first_step_statistics(loss, raw, full):
    raw_norm, full_norm = squared_norm(raw).sqrt(), squared_norm(full).sqrt()
    dot = sum((g.double() * p.double()).sum() for g, p in zip(raw, full, strict=True))
    return dict(support_loss_start=loss.item(),
                norm_ratio_step1=safe_ratio(full_norm, raw_norm).item(),
                cos_g_pg_step1=safe_ratio(dot, raw_norm * full_norm).item())


@torch.enable_grad()
def adapt_variant(learner, support, labels, ways, variant, gamma, steps=None):
    """Same computation as helpers.adapt(), with the encoder update replaced.

    Returns (fast weights or None if the support loss became nonfinite,
    first-step statistics or None). Weights are detached after every step; the
    values equal the baseline's because the baseline's update gradients are
    first-order (create_graph=False) as well.
    """
    model, transport = learner.learner, learner.transport
    config = learner.config["method_config"]
    weights = list(model.parameters())
    if transport.names != [name for name, _ in model.named_parameters()]:
        raise ValueError("LRSG parameter names must match the encoder ordering")
    with torch.no_grad():
        features = model.forward_weights(support, weights, embedding=True)
        head = prototype_head(features, labels, ways)
        conditioning = transport.condition(features.mean(dim=0).detach())
    fast = ([w.detach().clone().requires_grad_() for w in weights]
            + [h.detach().requires_grad_() for h in head])
    rates = [config["encoder_lr"]] * len(weights) + [config["classifier_lr"]] * 2
    clip = config["grad_clip"]
    statistics = None
    for step in range(config["inner_steps"] if steps is None else steps):
        loss = F.cross_entropy(model.forward_weights(support, fast), labels)
        if not torch.isfinite(loss):
            return None, statistics
        grads = torch.autograd.grad(loss, fast)
        if clip is not None:
            grads = [g.clamp(-clip, clip) for g in grads]
        with torch.no_grad():
            raw = [g.detach() for g in grads[:len(weights)]]
            if variant == "plain":
                encoder = raw
            else:
                full = [transport.transport_gradient(name, g, conditioning, gamma=gamma)
                        for name, g in zip(transport.names, raw, strict=True)]
                if step == 0:
                    statistics = first_step_statistics(loss, raw, full)
                encoder = variant_gradients(variant, raw, full)
            updates = list(encoder) + list(grads[len(weights):])
            fast = [(w - lr * g).requires_grad_()
                    for w, lr, g in zip(fast, rates, updates, strict=True)]
    return [w.detach() for w in fast], statistics


def self_check(learner, support, labels, ways, gamma):
    """full/plain must equal the unchanged baseline adapt() with/without transport."""
    model, transport = learner.learner, learner.transport
    config = dict(learner.config["method_config"], eval_gamma=gamma)
    worst = 0.0
    for variant, baseline_transport in (("full", transport), ("plain", None)):
        expected = adapt(model, list(model.parameters()), support, labels, config, ways,
                         baseline_transport, phase="test")
        actual, _ = adapt_variant(learner, support, labels, ways, variant, gamma)
        if actual is None:
            raise RuntimeError(f"Self-check: {variant} diverged")
        for a, e in zip(actual, expected, strict=True):
            e = e.detach()
            worst = max(worst, (a - e).abs().max().item())
            if not torch.allclose(a, e, rtol=SELF_CHECK_RTOL, atol=SELF_CHECK_ATOL):
                raise RuntimeError(f"Self-check failed: {variant} differs from baseline "
                                   f"adapt() (max abs diff {worst:g})")
    return worst


def evaluate_task(learner, task, task_id, gamma, check=False):
    # Verify parameters/buffers remain untouched and preserve the caller's RNG.
    with observational_diagnostics(learner):
        return _evaluate_task(learner, task, task_id, gamma, check)


def _evaluate_task(learner, task, task_id, gamma, check):
    support_x, support_y, _ = task.support_set
    query_x, query_y, _ = task.query_set
    support, labels = support_x.to(learner.dev), support_y.to(learner.dev)
    truth = query_y.cpu().numpy()
    row = dict(task_id=task_id, dataset=task.dataset,
               num_ways=task.num_ways, num_shots=task.num_shots)

    def accuracy(label, variant, steps=None, may_diverge=False):
        fast, statistics = adapt_variant(learner, support, labels, task.num_ways,
                                         variant, gamma, steps)
        probabilities = None
        if fast is not None:
            probabilities = MyPredictor(learner.learner, fast, learner.dev).predict(query_x)
        diverged = probabilities is None or not np.isfinite(probabilities).all()
        if diverged:
            if not may_diverge:
                raise ValueError(f"Task {task_id}: nonfinite predictions at {label}")
            probabilities = np.full((len(truth), task.num_ways), 1.0 / task.num_ways)
        return scorer_accuracy(probabilities, truth), int(diverged), statistics

    row["acc_steps0"], _, _ = accuracy("steps0", "plain", steps=0)
    statistics = None
    for variant in VARIANTS:
        counterfactual = variant in COUNTERFACTUALS
        row[f"acc_{variant}"], diverged, current = accuracy(variant, variant,
                                                            may_diverge=counterfactual)
        if counterfactual:
            row[f"diverged_{variant}"] = diverged
        else:
            statistics = current
    if statistics is None:  # inner_steps == 0: nothing to decompose
        raise ValueError("Direction x magnitude decomposition requires inner_steps >= 1")
    row.update(statistics)
    for name in STEP1_FIELDS:
        if not math.isfinite(row[name]):
            raise ValueError(f"Task {task_id}: nonfinite diagnostic {name}={row[name]}")
    row = {field: row[field] for field in FIELDS}  # CSV column order
    if check:
        row["_self_check_max_abs_diff"] = self_check(learner, support, labels,
                                                     task.num_ways, gamma)
    return row


def print_summary(sums):
    groups = [g for g in ["ALL"] + [label for label, _, _ in SHOT_BINS] if sums.get(g)]

    def mean(group, field):
        return sums[group][field] / sums[group]["n"]

    print("\nMean accuracy in %, tasks weighted equally. Rows: all tasks, then shot bins.")
    print("group".ljust(8) + "n".rjust(7)
          + "".join(c.replace("acc_", "").rjust(12) for c in ACCURACY_FIELDS))
    for group in groups:
        print(group.ljust(8) + f"{int(sums[group]['n'])}".rjust(7)
              + "".join(f"{100 * mean(group, c):.3f}".rjust(12) for c in ACCURACY_FIELDS))
    print("\nPaired differences in percentage points (first minus second).")
    labels = [f"{a}-{b}" for a, b in CONTRASTS]
    width = max(len(label) for label in labels) + 2
    print("group".ljust(8) + "".join(label.rjust(width) for label in labels))
    for group in groups:
        print(group.ljust(8) + "".join(
            f"{100 * (mean(group, f'acc_{a}') - mean(group, f'acc_{b}')):+.3f}".rjust(width)
            for a, b in CONTRASTS))
    print("\nDiverged tasks in % (scored at chance) and shared first-step statistics.")
    print("group".ljust(8) + "".join(v.rjust(12) for v in COUNTERFACTUALS)
          + "".join(s.rjust(20) for s in STEP1_FIELDS))
    for group in groups:
        print(group.ljust(8)
              + "".join(f"{100 * mean(group, f'diverged_{v}'):.2f}".rjust(12)
                        for v in COUNTERFACTUALS)
              + "".join(f"{mean(group, s):.4g}".rjust(20) for s in STEP1_FIELDS))
    print("\nDiagnostic of one trained checkpoint on meta-test tasks; the variants "
          "are not separately trained methods.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--reference_task_results")
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--test_tasks_per_dataset", type=int, default=600)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Multiplier of the saved lrsg.beta inside Pg; the reference "
                             "task_results.csv must have been scored at the same gamma")
    parser.add_argument("--self_check_tasks", type=int, default=5,
                        help="Compare full/plain with the baseline adapt() on the first N tasks")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32 or args.image_size < 1 or args.test_tasks_per_dataset < 1:
        parser.error("Require uint32 --seed, positive --image_size and --test_tasks_per_dataset")
    if not math.isfinite(args.gamma) or args.gamma < 0 or args.self_check_tasks < 0:
        parser.error("Require finite non-negative --gamma and non-negative --self_check_tasks")
    if args.reference_task_results and Path(args.out_csv).resolve() == Path(args.reference_task_results).resolve():
        parser.error("--out_csv must not overwrite --reference_task_results")
    if Path(args.out_csv).exists():
        parser.error("--out_csv must be a new file; refusing to overwrite an existing file")

    learner = MyLearner()
    learner.load(args.checkpoint)
    validate_checkpoint_config(learner.config)
    print(f"Loaded LR-only-TC checkpoint {args.checkpoint}; saved beta={learner.transport.beta:g}; "
          f"gamma={args.gamma:g}; inner_steps={learner.config['method_config']['inner_steps']}; "
          f"variants={list(VARIANTS)}", flush=True)
    # Identical to eval_beta_sweep.py and normal ingestion: the CLI seed controls
    # BOTH dataset preparation and episode sampling.
    _, _, test_info = prepare_datasets_information(
        args.input_data_dir, learner.config["validation_datasets"], args.seed, False)
    datasets = create_datasets(test_info, args.image_size)
    expected_count = len(datasets) * args.test_tasks_per_dataset
    if not expected_count:
        raise ValueError("No meta-test datasets available")
    loader = CompetitionDataLoader(datasets, TEST_EPISODES, args.seed, test_generator=True)
    reference = (ReferenceTaskResults(args.reference_task_results, expected_count)
                 if args.reference_task_results else None)
    if reference is None:
        print("Reference parity UNVERIFIED: provide --reference_task_results before "
              "interpreting the decomposition.", flush=True)
    sums = defaultdict(lambda: defaultdict(float))
    count, worst = 0, 0.0
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with new_output_csv(args.out_csv) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for count, task in enumerate(loader.generator(args.test_tasks_per_dataset), 1):
            row = evaluate_task(learner, task, count, args.gamma,
                                check=count <= args.self_check_tasks)
            worst = max(worst, row.pop("_self_check_max_abs_diff", 0.0))
            if count == args.self_check_tasks:
                print(f"Self-check PASSED on {count} tasks: full/plain match baseline adapt() "
                      f"(max abs weight diff {worst:g})", flush=True)
            if reference is not None:
                # ReferenceTaskResults compares its accuracy column with acc_steps5.
                reference.check(dict(row, acc_steps5=row["acc_full"]))
            writer.writerow(row)
            for group in ("ALL", shot_bin(task.num_shots)):
                sums[group]["n"] += 1
                for field in FIELDS[len(ID_FIELDS):]:
                    sums[group][field] += row[field]
            if count % 100 == 0:
                print(f"{count}/{expected_count} tasks done ({task.dataset})", flush=True)
        if count != expected_count:
            raise ValueError(f"Generated task count mismatch: expected {expected_count}, got {count}")
        if reference is not None:
            reference.finish()
    if reference is not None:
        print(f"Reference parity PASSED (full, gamma={args.gamma:g}): {count} tasks, "
              f"abs_tol={PARITY_ATOL:g}, rel_tol=0")
    print_summary(sums)
    print(f"\nPer-task rows written to {args.out_csv}")


if __name__ == "__main__":
    main()
