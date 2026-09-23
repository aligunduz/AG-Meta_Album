"""Independent FO-Proto-ConstZ-LRSGMAML submission using the repository FOMAML protocol."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from api import MetaLearner, Learner, Predictor
from network import ResNet
from helpers_fo_proto_constz_lrsgmaml import adapt, query_loss
from task_transport import ConstantConditionedTransport, validate_constant_condition
from metrics import log_metrics

SEED = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def read_config():
    return json.loads(Path(__file__).with_name("config.json").read_text())


def validate_config(config):
    validate_constant_condition(config)
    method = config["method_config"]
    if config["method"] != "fo-proto-constz-lrsgmaml" or method["first_order"] is not True:
        raise ValueError("This baseline requires fo-proto-constz-lrsgmaml and first_order=true")
    for key in ("reset_classifier",):
        if method[key] is not False:
            raise ValueError(f"FO-Proto-ConstZ-LRSGMAML requires {key}=false")
    for key in ("gradient_transport", "low_rank_transport"):
        if method[key] is not config["lrsg"]["enabled"]:
            raise ValueError(f"{key} must match lrsg.enabled")
    if method["task_conditioned_gate"] is not config["task_conditioning"]["enabled"]:
        raise ValueError("task_conditioned_gate must match task_conditioning.enabled")
    if config["test_checkpoint"] != "max-va.pth":
        raise ValueError("Test must use max-va.pth")


def snapshot(encoder, transport):
    def cpu_state(module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    return dict(encoder=cpu_state(encoder), lrsg=cpu_state(transport),
                architecture=transport.architecture())


def make_encoder(args):
    model = ResNet(**args).to(args["dev"])
    # The inherited functional forward accepts a task-local W,b. No permanent
    # classifier is registered, optimized, or saved by this baseline.
    model.model["out"] = nn.Identity()
    return model


class MyMetaLearner(MetaLearner):
    def __init__(self, train_classes, total_classes, logger):
        super().__init__(train_classes, total_classes, logger)
        self.config = read_config()
        validate_config(self.config)
        self.params = self.config["method_config"]
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_args = dict(num_classes=train_classes or 1, dev=self.dev,
                               num_blocks=18, pretrained=False)
        self.meta_learner = make_encoder(self.model_args)
        self.weights = list(self.meta_learner.parameters())
        self.transport = ConstantConditionedTransport(self.meta_learner, self.config)
        # Includes z only in zero mode; EMA buffers are never optimized.
        self.meta_parameters = self.weights + list(self.transport.parameters())
        self.logger = logger
        self.optimizer = torch.optim.Adam(self.meta_parameters, lr=self.params["outer_lr"])
        self.best_score = -float("inf")
        self.best_state = None

    def meta_fit(self, meta_train_generator, meta_valid_generator):
        exp = self.config["experiment_config"]
        buffer = [torch.zeros_like(w) for w in self.meta_parameters]
        for i, task in enumerate(meta_train_generator(exp["train_iterations"])):
            self.meta_learner.train()
            self.transport.train()
            support, labels, _ = task.support_set
            query, targets, _ = task.query_set
            use_history = self.transport.init_mode in ("ema", "shuffle")
            adapted = adapt(self.meta_learner, self.weights, support.to(self.dev),
                            labels.to(self.dev), self.params, task.num_ways, self.transport,
                            return_task_embedding=use_history)
            fast, task_embedding = adapted if use_history else (adapted, None)
            out, loss = query_loss(self.meta_learner, fast, query.to(self.dev),
                                   targets.to(self.dev))
            loss.backward()
            # Preserve FOMAML's per-task clipping and SUM over the meta-batch.
            for j, w in enumerate(self.meta_parameters):
                grad = w.grad
                if grad is None:
                    raise RuntimeError("Outer parameter disconnected from query loss")
                if self.params["grad_clip"] is not None:
                    grad = grad.clamp(-self.params["grad_clip"], self.params["grad_clip"])
                buffer[j].add_(grad)
            self.optimizer.zero_grad(set_to_none=True)
            if (i + 1) % self.params["meta_batch_size"] == 0:
                for w, grad in zip(self.meta_parameters, buffer):
                    w.grad = grad
                self.optimizer.step()
                buffer = [torch.zeros_like(w) for w in self.meta_parameters]
                self.optimizer.zero_grad(set_to_none=True)
            # Update once per completed training episode, including episodes
            # within a meta-batch, and before validation/checkpoint selection.
            if use_history:
                self.transport.complete_training_episode(task_embedding)
            self.log(task, out.detach().cpu().numpy(), loss.item())
            if (i + 1) % exp["validate_every"] == 0:
                log_metrics(self.logger, self.transport, i + 1)
                self.meta_valid(meta_valid_generator)
        if self.transport._count:
            log_metrics(self.logger, self.transport, exp["train_iterations"])
        # Never silently label an unvalidated last snapshot as best-validation.
        if self.best_state is None:
            self.meta_valid(meta_valid_generator)
        return MyLearner(self.model_args, self.best_state, self.config, self.best_score)

    @torch.no_grad()
    def meta_valid(self, generator):
        self.meta_learner.eval()
        self.transport.eval()
        correct = total = 0
        for task in generator(self.config["experiment_config"]["validation_tasks"]):
            support, labels, _ = task.support_set
            query, targets, _ = task.query_set
            fast = adapt(self.meta_learner, self.weights, support.to(self.dev),
                         labels.to(self.dev), self.params, task.num_ways, self.transport)
            out, _ = query_loss(self.meta_learner, fast, query.to(self.dev),
                                targets.to(self.dev))
            correct += (out.argmax(1).cpu() == targets.cpu()).sum().item()
            total += targets.numel()
            self.log(task, out.cpu().numpy(), meta_train=False)
        if not total:
            raise ValueError("Best-checkpoint selection requires validation episodes")
        score = correct / total
        if score > self.best_score:
            self.best_score = score
            self.best_state = snapshot(self.meta_learner, self.transport)


class MyLearner(Learner):
    def __init__(self, model_args=None, state=None, config=None, best_score=None):
        super().__init__()
        self.model_args, self.state = model_args, state
        self.config, self.best_score = config, best_score
        if state is not None:
            self._initialize()

    def _initialize(self):
        validate_config(self.config)
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_args = dict(self.model_args, dev=self.dev)
        self.learner = make_encoder(self.model_args)
        self.learner.load_state_dict(self.state["encoder"])
        self.transport = ConstantConditionedTransport(self.learner, self.config)
        if self.transport.architecture() != self.state["architecture"]:
            raise ValueError("Checkpoint ConstZ-LRSG architecture mismatch")
        self.transport.load_state_dict(self.state["lrsg"], strict=True)
        self.transport.eval()
        self.learner.eval()

    def fit(self, support_set):
        support, labels, _, ways, _ = support_set
        fast = adapt(self.learner, list(self.learner.parameters()),
                     support.to(self.dev), labels.to(self.dev),
                     self.config["method_config"], ways, self.transport)
        return MyPredictor(self.learner, fast, self.dev)

    def save(self, path_to_save):
        path = Path(path_to_save)
        if not path.is_dir():
            raise ValueError("Checkpoint directory must exist")
        torch.save(dict(format_version=1, method="fo-proto-constz-lrsgmaml",
                        config=self.config, model_args=dict(self.model_args, dev="cpu"),
                        state=self.state, best_validation_accuracy=self.best_score),
                   path / "max-va.pth")

    def load(self, path_to_load):
        path = Path(path_to_load)
        checkpoint = path / "max-va.pth" if path.is_dir() else path
        if checkpoint.name != "max-va.pth":
            raise ValueError("Expected best-validation checkpoint max-va.pth")
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data["method"] != "fo-proto-constz-lrsgmaml" or data["format_version"] != 1:
            raise ValueError("Unsupported FO-Proto-ConstZ-LRSGMAML checkpoint")
        self.model_args, self.state = data["model_args"], data["state"]
        self.config = data["config"]
        self.best_score = data["best_validation_accuracy"]
        self._initialize()


class MyPredictor(Predictor):
    def __init__(self, model, weights, dev):
        super().__init__()
        self.model, self.weights, self.dev = model, weights, dev

    @torch.no_grad()
    def predict(self, query_set):
        return self.model.forward_weights(query_set.to(self.dev), self.weights).softmax(1).cpu().numpy()
