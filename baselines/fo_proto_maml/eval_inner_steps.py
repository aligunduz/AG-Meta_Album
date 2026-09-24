"""Paired inner-loop ablation for one FO-Proto-MAML checkpoint.

Every meta-test episode of the runner's protocol (same seed, same datasets,
same 2-20-way / 1-20-shot sampling) is evaluated with the SAME encoder and
several inner-loop lengths (default 0 and 5). inner_steps=0 is the pure
prototype classifier, so acc_steps5 - acc_steps0 is the contribution of the
inner loop alone, free of any training-run difference.

Per task it also records the support loss at the prototype start, the norm of
the first encoder support gradient (head held fixed, as in the inner loop) and
the support loss after the longest adaptation. One CSV row per task.

Example (repository root):
    python -u baselines/fo_proto_maml/eval_inner_steps.py \
        --checkpoint=/path/to/model/max-va.pth \
        --input_data_dir=/content/meta_album_feedback \
        --out_csv=/path/to/inner_steps_0_vs_5.csv
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model import MyLearner  # noqa: E402
from helpers_fo_proto_maml import prototype_head  # noqa: E402
from cdmetadl.helpers.general_helpers import prepare_datasets_information  # noqa: E402
from cdmetadl.ingestion.image_dataset import create_datasets  # noqa: E402
from cdmetadl.ingestion.data_generator import CompetitionDataLoader  # noqa: E402

TEST_EPISODES = dict(N=None, min_N=2, max_N=20, k=None, min_k=1, max_k=20,
                     query_images_per_class=20)


def support_diagnostics(learner, support, labels, ways, adapted):
    """Support loss and encoder-gradient norm at the prototype start, and the
    support loss after adaptation with the given fast weights."""
    model = learner.learner
    weights = [w.detach() for w in model.parameters()]
    with torch.no_grad():
        features = model.forward_weights(support, weights, embedding=True)
        head = list(prototype_head(features, labels, ways))
    with torch.enable_grad():
        body = [w.clone().requires_grad_() for w in weights]
        loss_start = F.cross_entropy(model.forward_weights(support, body + head), labels)
        grads = torch.autograd.grad(loss_start, body)
    grad_norm = torch.sqrt(sum((g.double() ** 2).sum() for g in grads)).item()
    with torch.no_grad():
        loss_after = F.cross_entropy(
            model.forward_weights(support, [w.detach() for w in adapted]), labels)
    return loss_start.item(), grad_norm, loss_after.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--inner_steps", default="0,5",
                        help="comma-separated inner-loop lengths, e.g. 0,5")
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--test_tasks_per_dataset", type=int, default=100)
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=None,
        help="Optional test-time override for method_config encoder_lr"
    )
    args = parser.parse_args()
    steps = sorted({int(s) for s in args.inner_steps.split(",")})
    longest = steps[-1]

    learner = MyLearner()
    learner.load(args.checkpoint)
    base_config = dict(learner.config["method_config"])
    if args.encoder_lr is not None:
        base_config["encoder_lr"] = args.encoder_lr
    print(f"Loaded {args.checkpoint} (best val {learner.best_score:.4f}); "
          f"trained inner_steps={base_config['inner_steps']}; "
          f"encoder_lr={base_config['encoder_lr']}; evaluating {steps}")

    _, _, test_info = prepare_datasets_information(
        args.input_data_dir, learner.config["validation_datasets"], args.seed, False)
    loader = CompetitionDataLoader(create_datasets(test_info, args.image_size),
                                   TEST_EPISODES, args.seed, test_generator=True)

    fields = (["task_id", "dataset", "num_ways", "num_shots"]
              + [f"acc_steps{s}" for s in steps]
              + ["support_loss_start", "encoder_grad_norm_start",
                 f"support_loss_after{longest}"])
    sums = defaultdict(lambda: defaultdict(float))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, task in enumerate(loader.generator(args.test_tasks_per_dataset)):
            support_x, support_y, _ = task.support_set
            query_x, query_y, _ = task.query_set
            support_set = (support_x, support_y, None, task.num_ways, task.num_shots)
            row = dict(task_id=i + 1, dataset=task.dataset,
                       num_ways=task.num_ways, num_shots=task.num_shots)
            adapted = None
            for s in steps:
                learner.config["method_config"] = dict(base_config, inner_steps=s)
                predictor = learner.fit(support_set)
                predicted = predictor.predict(query_x).argmax(1)
                row[f"acc_steps{s}"] = float((predicted == query_y.numpy()).mean())
                if s == longest:
                    adapted = predictor.weights
            learner.config["method_config"] = dict(base_config)
            start, grad_norm, after = support_diagnostics(
                learner, support_x.to(learner.dev), support_y.to(learner.dev),
                task.num_ways, adapted)
            row.update(support_loss_start=start, encoder_grad_norm_start=grad_norm)
            row[f"support_loss_after{longest}"] = after
            writer.writerow(row)
            for name in ("ALL", task.dataset):
                sums[name]["n"] += 1
                for s in steps:
                    sums[name][s] += row[f"acc_steps{s}"]
            if (i + 1) % 100 == 0:
                print(f"{i + 1} tasks done ({task.dataset})", flush=True)

    print("\nMean query accuracy (%) per inner-loop length:")
    print("dataset".ljust(10) + "".join(f"steps{s}".rjust(10) for s in steps)
          + (f"{'diff':>10}" if len(steps) > 1 else ""))
    for name, acc in sums.items():
        means = [100 * acc[s] / acc["n"] for s in steps]
        diff = f"{means[-1] - means[0]:+10.2f}" if len(steps) > 1 else ""
        print(name.ljust(10) + "".join(f"{m:10.2f}" for m in means) + diff)
    print(f"\nPer-task rows written to {args.out_csv}")


if __name__ == "__main__":
    main()
