"""Keep runner episode logging; add interval aggregates and optional active W&B."""
import json
import sys
from pathlib import Path


def log_metrics(logger, transport, iteration):
    values = transport.metrics()
    transport.reset_metrics()
    if not values:
        return
    record = dict(iteration=iteration, **values)
    if getattr(logger, "logs_dir", None) is not None:
        directory = Path(logger.logs_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "lrsg_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(record))
    # The reference runner has no W&B dependency. Reuse an active run only;
    # never initialize a run or change its configuration automatically.
    wandb = sys.modules.get("wandb")
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.log(record)
