"""Evaluate max-va.pth with the runner's unchanged test episode protocol."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model import MyLearner
from cdmetadl.helpers.general_helpers import prepare_datasets_information
from cdmetadl.ingestion.image_dataset import create_datasets
from cdmetadl.ingestion.data_generator import CompetitionDataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--test_tasks_per_dataset", type=int, default=100)
    args = parser.parse_args()
    learner = MyLearner()
    learner.load(args.checkpoint)
    _, _, info = prepare_datasets_information(
        args.input_data_dir, learner.config["validation_datasets"], args.seed, False)
    loader = CompetitionDataLoader(
        create_datasets(info, args.image_size),
        dict(N=None, min_N=2, max_N=20, k=None, min_k=1, max_k=20,
             query_images_per_class=20), args.seed, test_generator=True)
    correct = total = 0
    for task in loader.generator(args.test_tasks_per_dataset):
        predictor = learner.fit((*task.support_set, task.num_ways, task.num_shots))
        predictions = predictor.predict(task.query_set[0]).argmax(1)
        labels = task.query_set[1].cpu().numpy()
        correct += (predictions == labels).sum()
        total += len(labels)
    if not total:
        raise ValueError("No test examples")
    print(f"Best-validation checkpoint query accuracy: {correct / total:.6f} ({total} images)")


if __name__ == "__main__":
    main()
