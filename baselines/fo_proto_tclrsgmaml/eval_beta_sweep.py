r"""Test-time scale sweep of the low-rank correction for an LR-only-TC checkpoint.

Encoder update per inner step (the checkpoint itself is never modified)::

    G~ = sigmoid(a) * G  +  gamma * beta * U diag(1 + delta_c) V^T G

gamma multiplies the saved ``lrsg.beta``, i.e. the WHOLE low-rank correction;
``delta_c`` and the scalar gates are untouched and classifier-head updates are
not scaled. Reading the columns:

* ``acc_gamma_1`` is the legacy evaluation policy and must reproduce a legacy
  gamma=1 scorer's ``task_results.csv`` (reference parity, abs_tol=1e-12).
* ``acc_gamma_2`` matches the baseline's new default evaluation policy.
* ``acc_gamma_0`` keeps the LRTC-trained encoder and its learned scalar gates
  and only switches the low-rank correction off. It is NOT FO-Proto-MAML.
* ``acc_steps0`` is the prototype start of the same encoder (no adaptation).

Every gamma adapts the same task object (identical support/query tensors),
restarting from the checkpoint weights. Predictions use the ingestion/scorer's
six-decimal serialization before argmax, including its first-index tie rule.
The output must be a new file; a failed evaluation removes its incomplete CSV.

Run from the repository root, using the original run's data, seed and image size::

    python baselines/fo_proto_tclrsgmaml/eval_beta_sweep.py \
        --checkpoint /path/to/model/max-va.pth \
        --input_data_dir /content/meta_album_final --seed 93 \
        --image_size 128 --test_tasks_per_dataset 600 \
        --gammas 0,0.25,0.5,1,2,4 \
        --reference_task_results /path/to/scoring/task_results.csv \
        --out_csv /path/to/lrtc_seed93_gamma_sweep.csv

Accuracies in the CSV are fractions; the printed summary uses percent and
weights tasks equally. Choosing gamma on these meta-test tasks is exploratory:
the accuracy of a gamma selected here is not independent performance evidence.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import io
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MyLearner  # noqa: E402
from eval_inner_steps import (PARITY_ATOL, TEST_EPISODES, ReferenceTaskResults,  # noqa: E402
                              observational_diagnostics, validate_checkpoint_config)
from cdmetadl.helpers.general_helpers import prepare_datasets_information  # noqa: E402
from cdmetadl.ingestion.image_dataset import create_datasets  # noqa: E402
from cdmetadl.ingestion.data_generator import CompetitionDataLoader  # noqa: E402

DEFAULT_GAMMAS = "0,0.25,0.5,1,2,4"
ID_FIELDS = ("task_id", "dataset", "num_ways", "num_shots")
SHOT_BINS = (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11-20", 11, 20))


def gamma_field(gamma):
    return f"acc_gamma_{gamma:g}"


def parse_gammas(text):
    """Comma-separated, finite, non-negative, unique; must contain 1 for parity."""
    try:
        gammas = tuple(float(item) for item in text.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid --gammas {text!r}") from exc
    if not gammas or any(not math.isfinite(g) or g < 0 for g in gammas):
        raise ValueError("--gammas must be finite and non-negative")
    if len({gamma_field(g) for g in gammas}) != len(gammas):
        raise ValueError("--gammas must be unique")
    if 1.0 not in gammas:
        raise ValueError("--gammas must contain 1 (the trained model, used for reference parity)")
    return gammas


@contextmanager
def scaled_low_rank(transport, gamma):
    """Temporarily use beta * gamma; the saved beta is restored even on failure."""
    saved = transport.beta
    scaled = saved * gamma
    if not math.isfinite(gamma) or gamma < 0 or not math.isfinite(scaled):
        raise ValueError("gamma and scaled beta must be finite; gamma must be non-negative")
    transport.beta = scaled
    try:
        yield
    finally:
        transport.beta = saved


def scorer_accuracy(probabilities, truth):
    """Match ingestion's np.savetxt(fmt='%f') and the scorer's np.loadtxt."""
    with io.StringIO() as stream:
        np.savetxt(stream, probabilities, fmt="%f")
        stream.seek(0)
        serialized = np.loadtxt(stream, ndmin=2)
    return float((serialized.argmax(1) == truth).mean())


def evaluate_task(learner, task, task_id, gammas):
    # Functional adapt clones the loaded weights on every fit. The repository's
    # functional batchnorm uses fresh statistics, even with the model in eval.
    # Verify parameters/buffers remain untouched and preserve the caller's RNG.
    with observational_diagnostics(learner):
        return _evaluate_task(learner, task, task_id, gammas)


def _evaluate_task(learner, task, task_id, gammas):
    support_x, support_y, _ = task.support_set
    query_x, query_y, _ = task.query_set
    support_set = (support_x, support_y, None, task.num_ways, task.num_shots)
    truth = query_y.cpu().numpy()
    row = dict(task_id=task_id, dataset=task.dataset,
               num_ways=task.num_ways, num_shots=task.num_shots)

    def accuracy(label):
        predictor = learner.fit(support_set)  # always restarts from checkpoint weights
        probabilities = predictor.predict(query_x)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Task {task_id}: nonfinite predictions at {label}")
        return scorer_accuracy(probabilities, truth)

    original_config = learner.config["method_config"]
    try:
        learner.config["method_config"] = dict(original_config, inner_steps=0)
        row["acc_steps0"] = accuracy("steps0")
    finally:
        learner.config["method_config"] = original_config
    try:
        # This experiment already scales beta. Disable the normal evaluation
        # multiplier so each requested gamma is applied exactly once.
        learner.config["method_config"] = dict(original_config, eval_gamma=1.0)
        for gamma in gammas:
            with scaled_low_rank(learner.transport, gamma):
                row[gamma_field(gamma)] = accuracy(f"gamma={gamma:g}")
    finally:
        learner.config["method_config"] = original_config
    return row


def shot_bin(shots):
    for label, low, high in SHOT_BINS:
        if low <= shots <= high:
            return label
    raise ValueError(f"num_shots={shots} outside the meta-test range")


def print_summary(sums, columns):
    print("\nMean accuracy in %, tasks weighted equally. Rows: all tasks, then shot bins.")
    print("group".ljust(8) + "n".rjust(7) + "".join(c.replace("acc_", "").rjust(12) for c in columns))
    for group in ["ALL"] + [label for label, _, _ in SHOT_BINS]:
        values = sums.get(group)
        if not values:
            continue
        n = values["n"]
        print(group.ljust(8) + f"{int(n)}".rjust(7)
              + "".join(f"{100 * values[c] / n:.3f}".rjust(12) for c in columns))
    print("\nExploratory only: a gamma chosen on these meta-test tasks is not "
          "independent performance evidence.")


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
    parser.add_argument("--gammas", default=DEFAULT_GAMMAS,
                        help="Comma-separated multipliers of the saved lrsg.beta; must contain 1")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32 or args.image_size < 1 or args.test_tasks_per_dataset < 1:
        parser.error("Require uint32 --seed, positive --image_size and --test_tasks_per_dataset")
    if args.reference_task_results and Path(args.out_csv).resolve() == Path(args.reference_task_results).resolve():
        parser.error("--out_csv must not overwrite --reference_task_results")
    if Path(args.out_csv).exists():
        parser.error("--out_csv must be a new file; refusing to overwrite an existing file")
    try:
        gammas = parse_gammas(args.gammas)
    except ValueError as exc:
        parser.error(str(exc))
    columns = ("acc_steps0",) + tuple(gamma_field(g) for g in gammas)
    fields = ID_FIELDS + columns

    learner = MyLearner()
    learner.load(args.checkpoint)
    validate_checkpoint_config(learner.config)
    saved_beta = learner.transport.beta
    print(f"Loaded LR-only-TC checkpoint {args.checkpoint}; saved beta={saved_beta:g}; "
          f"inner_steps={learner.config['method_config']['inner_steps']}; "
          f"gammas={[f'{g:g}' for g in gammas]}", flush=True)
    # Identical to eval_inner_steps.py and normal ingestion: the CLI seed controls
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
              "interpreting the sweep.", flush=True)
    sums = defaultdict(lambda: defaultdict(float))
    count = 0
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with new_output_csv(args.out_csv) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for count, task in enumerate(loader.generator(args.test_tasks_per_dataset), 1):
            row = evaluate_task(learner, task, count, gammas)
            if reference is not None:
                # ReferenceTaskResults compares its accuracy column with acc_steps5.
                reference.check(dict(row, acc_steps5=row[gamma_field(1.0)]))
            writer.writerow(row)
            for group in ("ALL", shot_bin(task.num_shots)):
                sums[group]["n"] += 1
                for column in columns:
                    sums[group][column] += row[column]
            if count % 100 == 0:
                print(f"{count}/{expected_count} tasks done ({task.dataset})", flush=True)
        if count != expected_count:
            raise ValueError(f"Generated task count mismatch: expected {expected_count}, got {count}")
        if reference is not None:
            reference.finish()
        if learner.transport.beta != saved_beta:
            raise RuntimeError("lrsg.beta was not restored")
    if reference is not None:
        print(f"Reference parity PASSED (gamma=1): {count} tasks, abs_tol={PARITY_ATOL:g}, rel_tol=0")
    print_summary(sums, columns)
    print(f"\nPer-task rows written to {args.out_csv}")


@contextmanager
def new_output_csv(path):
    """Exclusive creation also protects existing checkpoints and hard links."""
    with open(path, "x", newline="", encoding="utf-8") as handle:
        try:
            yield handle
        except BaseException:
            # This file was exclusively created by this call, so no previous
            # result is removed. Close first to allow unlink on Windows.
            handle.close()
            Path(path).unlink()
            raise


if __name__ == "__main__":
    main()
