"""Script-owned learner adapter; unchanged native ingestion is unsupported."""
from pathlib import Path

from baselines.fo_proto_domain_ema_stage2.data import DATASET_DOMAIN, canonical
from baselines.fo_proto_domain_ema_stage2.experiment import Experiment
from baselines.fo_proto_domain_ema_stage2.source import reference


class Stage2Learner:
    def __init__(self, experiment=None):
        self.experiment = experiment
        self._context = None
        self._last_context = None
        self.selected_branch = None

    def set_task_context(self, context):
        if not context or "dataset_id" not in context:
            raise ValueError("Stage2 requires explicit dataset task context")
        identifier = canonical(context["dataset_id"])
        domain = DATASET_DOMAIN[identifier]
        if context.get("domain") != domain:
            raise ValueError("Task dataset/domain context mismatch")
        self._context = dict(context, dataset_id=identifier, domain=domain)

    def fit(self, support_set):
        if len(support_set) != 5:
            raise ValueError("fit expects (x, y, original_y, ways, shots)")
        if self._context is None:
            raise ValueError("Call set_task_context for every task; no implicit routing")
        context, self._context = self._context, None
        key = self.experiment.route(context["domain"], "B")
        fast = self.experiment.adapt(support_set, key)
        self._last_context, self.selected_branch = context, key
        return reference().MyPredictor(self.experiment.encoder, fast, self.experiment.base.dev)

    def save(self, path_to_save):
        self.experiment.save(Path(path_to_save) / "stage2-final.pth")

    def load(self, path_to_load):
        path = Path(path_to_load)
        if path.is_dir():
            path = path / "stage2-final.pth"
        self.experiment = Experiment.load(path)
        self._context = None
