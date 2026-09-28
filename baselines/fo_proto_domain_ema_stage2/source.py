"""Load unchanged source files in private module namespaces, without patching.

The legacy baseline uses short absolute sibling imports. Resolve those only
inside its private module builtins, never by replacing sys.modules entries,
sys.path, builtins.__import__, or attributes of any shared module.
"""
import builtins
import random
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "fo_proto_constz_lrsgmaml"
_MODULE = None


def reference():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    names = ("api", "weight_names", "network", "gate_net", "low_rank_transport",
             "task_transport", "helpers_fo_proto_constz_lrsgmaml", "metrics", "model")
    modules = {}

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in names:
            return load_module(name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load_module(name):
        if name not in modules:
            path = ROOT / (name + ".py")
            module = ModuleType("_stage2_source_" + name)
            module.__file__ = str(path)
            module.__dict__["__builtins__"] = dict(vars(builtins), __import__=local_import)
            modules[name] = module
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        return modules[name]

    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            _MODULE = load_module("model")
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return _MODULE
