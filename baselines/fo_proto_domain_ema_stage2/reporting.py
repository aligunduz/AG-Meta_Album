"""Atomic writes for this run's artifacts; never replace unrelated outputs."""
import json
import math
import os
from pathlib import Path
import tempfile


def atomic_write(path, write, *, replace=False):
    path = Path(path)
    if not replace and path.exists():
        raise FileExistsError(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_summary(path, summary):
    # Callers own a freshly created run directory. Updates replace only this
    # run's pending/partial report, and keep the prior report on write failure.
    # Preserve failure reports even when a failing numeric check left NaN/Inf
    # in diagnostics. Strings explicitly mark these as invalid measurements.
    def serializable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return value

    payload = json.dumps(serializable(summary), indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda handle: handle.write(payload), replace=True)


def error_record(error):
    return dict(type=type(error).__name__, message=str(error))
