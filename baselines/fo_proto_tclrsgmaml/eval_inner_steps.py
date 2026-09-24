r"""Paired step-0/step-5 diagnostics for an existing LR-only-TC checkpoint.

Run from the repository root, using the original run's data, seed and image size::

    python baselines/fo_proto_tclrsgmaml/eval_inner_steps.py \
        --checkpoint /path/to/model/max-va.pth \
        --input_data_dir /content/meta_album_feedback --seed 93 \
        --image_size 128 --test_tasks_per_dataset 100 \
        --reference_task_results /path/to/scoring_output/task_results.csv \
        --out_csv /path/to/lrtc_seed93_inner_steps.csv

The saved checkpoint config is authoritative. Accuracies and adaptation gains
in CSV are fractions; summaries use percent and percentage points. All means
weight tasks equally. Reference parity is required before interpreting the
seed 93/94/95 experiments; omitting the reference explicitly leaves it UNVERIFIED.

Raw and clipped encoder norms are logged separately. encoder_grad_norm_start
aliases encoder_grad_norm_raw_start for comparison with FO-Proto-MAML.
The low-rank correction uses clipped gradients; correction_to_gradient_ratio_start
divides by encoder_grad_norm_clipped_start + 1e-12.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MyLearner  # noqa: E402
from helpers_fo_proto_tclrsgmaml import prototype_head  # noqa: E402
from cdmetadl.helpers.general_helpers import prepare_datasets_information  # noqa: E402
from cdmetadl.ingestion.image_dataset import create_datasets  # noqa: E402
from cdmetadl.ingestion.data_generator import CompetitionDataLoader  # noqa: E402

TEST_EPISODES = dict(N=None, min_N=2, max_N=20, k=None, min_k=1, max_k=20,
                     query_images_per_class=20)
METRICS = ("acc_steps0", "acc_steps5", "adaptation_gain", "support_loss_start",
           "encoder_grad_norm_start", "encoder_grad_norm_raw_start",
           "encoder_grad_norm_clipped_start", "support_loss_after5",
           "low_rank_correction_norm_start", "correction_to_gradient_ratio_start")
FIELDS = ("task_id", "dataset", "num_ways", "num_shots") + METRICS
PARITY_ATOL = 1e-12


def validate_checkpoint_config(config):
    """Reject other TC ablations without reading or modifying config.json."""
    expected = {
        ("method",): "fo-proto-tclrsgmaml",
        ("task_conditioning", "enabled"): True,
        ("task_conditioning", "scalar_delta_scale"): 0.0,
        ("task_conditioning", "low_rank_delta_scale"): 1.0,
        ("lrsg", "enabled"): True,
    }
    for path, wanted in expected.items():
        actual = config
        for key in path:
            actual = actual.get(key) if isinstance(actual, dict) else None
        valid = actual == wanted
        if isinstance(wanted, bool):
            valid = actual is wanted
        elif isinstance(wanted, float):
            valid = type(actual) in (int, float) and valid
        if not valid:
            raise ValueError(f"LR-only-TC checkpoint requires {'.'.join(path)}="
                             f"{wanted!r}; saved config has {actual!r}")


@contextmanager
def observational_diagnostics(learner):
    """Preserve RNG and verify all encoder/transport parameters and buffers."""
    modules = (learner.learner, learner.transport)
    if any(m.training for root in modules for m in root.modules()):
        raise ValueError("Diagnostics require encoder and transport in eval mode")
    tensors = [(f"{prefix}.{name}", value, value.detach().clone())
               for prefix, root in zip(("encoder", "transport"), modules)
               for name, value in list(root.named_parameters()) + list(root.named_buffers())]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [learner.dev.index or 0] if learner.dev.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for name, current, before in tensors:
            if not torch.equal(current.detach(), before):
                raise RuntimeError(f"Diagnostic mutated parameter/buffer: {name}")


def support_diagnostics(learner, support, labels, ways, adapted):
    """Measure the prototype start with a fixed head, as FO-Proto-MAML does.

    The encoder norm is over RAW gradients, before clipping or transport.
    The low-rank residual uses the actual transport input (after grad_clip),
    with the same dtype, projection and row scaling as LowRankTransport.
    Its ratio uses the clipped whole-encoder norm, including unranked tensors.
    encoder_grad_norm_start is the raw norm's FO-Proto-MAML compatibility alias.
    No scalar gate term or training metric accumulator enters these values.
    """
    with observational_diagnostics(learner):
        model, transport = learner.learner, learner.transport
        weights = [w.detach().clone() for w in model.parameters()]
        if transport.names != [name for name, _ in model.named_parameters()]:
            raise ValueError("Transport parameter ordering differs from encoder")
        with torch.no_grad():
            features = model.forward_weights(support, weights, embedding=True)
            head = list(prototype_head(features, labels, ways))
            conditioning = transport.condition(features.mean(dim=0).detach())
            if conditioning is None or not torch.equal(
                    conditioning[0], torch.zeros_like(conditioning[0])):
                raise ValueError("LR-only-TC requires conditioning with delta_a=0")
        with torch.enable_grad():
            body = [w.requires_grad_() for w in weights]
            loss_start = F.cross_entropy(model.forward_weights(support, body + head), labels)
            grads = torch.autograd.grad(loss_start, body)
        with torch.no_grad():
            grad_norm = torch.sqrt(sum(g.double().square().sum() for g in grads)).item()
            correction_squared = grads[0].new_zeros((), dtype=torch.float64)
            clip = learner.config["method_config"]["grad_clip"]
            clipped = [g if clip is None else g.clamp(-clip, clip) for g in grads]
            clipped_norm = torch.sqrt(sum(g.double().square().sum() for g in clipped)).item()
            for name, grad in zip(transport.names, clipped, strict=True):
                key = transport.indices[name]
                if key not in transport.u:
                    continue
                matrix = grad.reshape(grad.shape[0], -1)
                projected = transport.v[key].T @ matrix
                projected = (1 + conditioning[1][key]).unsqueeze(1) * projected
                correction = transport.beta * (transport.u[key] @ projected)
                correction_squared += correction.double().square().sum()
            correction_norm = correction_squared.sqrt().item()
            loss_after = F.cross_entropy(model.forward_weights(
                support, [w.detach() for w in adapted]), labels)
        result = dict(support_loss_start=loss_start.item(),
                      encoder_grad_norm_start=grad_norm,
                      encoder_grad_norm_raw_start=grad_norm,
                      encoder_grad_norm_clipped_start=clipped_norm,
                      support_loss_after5=loss_after.item(),
                      low_rank_correction_norm_start=correction_norm,
                      correction_to_gradient_ratio_start=correction_norm / (clipped_norm + 1e-12))
        require_finite(result)
        return result


def require_finite(values):
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite diagnostic: {name}={value}")


def evaluate_task(learner, task, task_id):
    support_x, support_y, _ = task.support_set
    query_x, query_y, _ = task.query_set
    support_set = (support_x, support_y, None, task.num_ways, task.num_shots)
    row = dict(task_id=task_id, dataset=task.dataset,
               num_ways=task.num_ways, num_shots=task.num_shots)
    original_config = learner.config["method_config"]
    try:
        for steps in (0, 5):
            learner.config["method_config"] = dict(original_config, inner_steps=steps)
            predictor = learner.fit(support_set)
            probabilities = predictor.predict(query_x)
            if not np.isfinite(probabilities).all():
                raise ValueError(f"Task {task_id}: nonfinite step-{steps} predictions")
            row[f"acc_steps{steps}"] = float(
                (probabilities.argmax(1) == query_y.cpu().numpy()).mean())
            if steps == 5:
                adapted = [w.detach() for w in predictor.weights]
            del predictor
    finally:
        # Restore the original object even when fit/predict raises.
        learner.config["method_config"] = original_config
    row["adaptation_gain"] = row["acc_steps5"] - row["acc_steps0"]
    row.update(support_diagnostics(learner, support_x.to(learner.dev),
                                   support_y.to(learner.dev), task.num_ways, adapted))
    require_finite({key: row[key] for key in METRICS})
    return row


class ReferenceTaskResults:
    """Strict ordered comparison with the scorer's unrounded accuracy column."""

    def __init__(self, path, expected_count):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"task_id", "dataset", "num_ways", "num_shots", "accuracy"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Reference task_results.csv missing columns: {sorted(missing)}")
            self.rows = list(reader)
        if len(self.rows) != expected_count:
            raise ValueError(f"Reference task count mismatch: expected {expected_count}, "
                             f"reference has {len(self.rows)}")
        self.checked = 0

    def check(self, row):
        index = self.checked
        if index >= len(self.rows):
            raise ValueError(f"Reference task count mismatch: extra task {row['task_id']}")
        reference = self.rows[index]
        for field in ("task_id", "dataset", "num_ways", "num_shots", "accuracy"):
            raw = reference[field]
            try:
                expected = (float(raw) if field == "accuracy" else
                            raw if field == "dataset" else int(raw))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Reference task {index + 1}: invalid {field}={raw!r}") from exc
            actual = row["acc_steps5"] if field == "accuracy" else row[field]
            matches = (math.isfinite(expected) and math.isfinite(actual)
                       and math.isclose(actual, expected, rel_tol=0, abs_tol=PARITY_ATOL)
                       if field == "accuracy" else actual == expected)
            if not matches:
                raise ValueError(f"Reference parity mismatch at task {row['task_id']} "
                                 f"({row['dataset']}), {field}: "
                                 f"diagnostic={actual!r}, reference={expected!r}")
        self.checked += 1

    def finish(self):
        if self.checked != len(self.rows):
            raise ValueError(f"Reference task count mismatch: evaluated {self.checked}, "
                             f"reference has {len(self.rows)}")


def print_summary(sums):
    print("\nTask means: accuracies in %, adaptation gain in percentage points.")
    for dataset, values in sums.items():
        means = {key: values[key] / values["n"] for key in METRICS}
        print(f"{dataset} (n={values['n']}): " + ", ".join(
            f"{key}={means[key] * (100 if key in METRICS[:3] else 1):.8g}"
            for key in METRICS))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--reference_task_results")
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--test_tasks_per_dataset", type=int, default=100)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32 or args.image_size < 1 or args.test_tasks_per_dataset < 1:
        parser.error("Require uint32 --seed, positive --image_size and --test_tasks_per_dataset")
    if args.reference_task_results and Path(args.out_csv).resolve() == Path(args.reference_task_results).resolve():
        parser.error("--out_csv must not overwrite --reference_task_results")

    learner = MyLearner()
    learner.load(args.checkpoint)
    validate_checkpoint_config(learner.config)
    print(f"Loaded LR-only-TC checkpoint {args.checkpoint}; "
          f"saved inner_steps={learner.config['method_config']['inner_steps']}; evaluating [0, 5]", flush=True)
    # Deliberately identical to the FO-Proto-MAML diagnostic and normal ingestion:
    # the CLI seed controls BOTH dataset preparation and episode sampling.
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
        print("Reference parity UNVERIFIED: provide --reference_task_results for each "
              "seed 93/94/95 experiment before interpreting diagnostics.", flush=True)
    sums = defaultdict(lambda: defaultdict(float))
    count = 0
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for count, task in enumerate(loader.generator(args.test_tasks_per_dataset), 1):
            row = evaluate_task(learner, task, count)
            if reference is not None:
                reference.check(row)
            writer.writerow(row)
            for dataset in ("ALL", task.dataset):
                sums[dataset]["n"] += 1
                for key in METRICS:
                    sums[dataset][key] += row[key]
            if count % 100 == 0:
                print(f"{count}/{expected_count} tasks done ({task.dataset})", flush=True)
        if count != expected_count:
            raise ValueError(f"Generated task count mismatch: expected {expected_count}, got {count}")
        if reference is not None:
            reference.finish()
    if reference is not None:
        print(f"Reference parity PASSED: {count} tasks, abs_tol={PARITY_ATOL:g}, rel_tol=0")
    print_summary(sums)
    print(f"\nPer-task rows written to {args.out_csv}")


if __name__ == "__main__":
    main()
