"""Train stage2 only; does not retrain stage1 or run the test split."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.fo_proto_domain_ema_stage2.data import loaders
from baselines.fo_proto_domain_ema_stage2.experiment import (
    Experiment, options_from, source_checkpoint, train)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_seed", type=int, required=True)
    parser.add_argument("--stage2_seed", type=int)
    parser.add_argument("--validation_seed", type=int)
    parser.add_argument("--test_seed", type=int)
    parser.add_argument("--budget_per_domain", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--source_provenance")
    parser.add_argument("--options", help="Optional overrides of config.json")
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    config.update({key: getattr(args, key) for key in
                   ("source_checkpoint", "data_seed", "stage2_seed", "validation_seed",
                    "test_seed", "budget_per_domain", "source_provenance")})
    options = options_from(config, args.options, args.data_seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source = source_checkpoint(options)
    (train_loader, valid_loader, _), manifest = loaders(args.input_data_dir, source["config"], options, args.image_size)
    experiment = Experiment(source, options, manifest)
    experiment.set_data_provenance(args.input_data_dir, args.image_size)
    print("Stage2 training domains:", experiment.seen, flush=True)
    train(experiment, train_loader.generator, valid_loader.generator, output)


if __name__ == "__main__":
    main()
