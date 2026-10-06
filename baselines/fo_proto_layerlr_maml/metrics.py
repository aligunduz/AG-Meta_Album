"""Keep runner episode logging; add learned step-size summaries and optional active W&B."""
import json
import sys
from pathlib import Path


def log_metrics(logger, inner_lrs, iteration):
    values = inner_lrs.metrics()
    record = dict(iteration=iteration, **values)
    if getattr(logger, "logs_dir", None) is not None:
        directory = Path(logger.logs_dir)
        directory.mkdir(parents=True, exist_ok=True)
        # The file keeps every per-tensor multiplier; stdout/W&B only the summary.
        with (directory / "inner_lr_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record, scales=inner_lrs.scales())) + "\n")
    print(json.dumps(record))
    # The reference runner has no W&B dependency. Reuse an active run only;
    # never initialize a run or change its configuration automatically.
    wandb = sys.modules.get("wandb")
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.log(record)
