"""Independent FO-Proto-MAML submission using the repository FOMAML protocol."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from api import MetaLearner, Learner, Predictor
from network import ResNet
from helpers_fo_proto_maml import adapt, query_loss

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
    method = config["method_config"]
    if config["method"] != "fo-proto-maml" or method["first_order"] is not True:
        raise ValueError("This baseline requires fo-proto-maml and first_order=true")
    for key in ("gradient_transport", "task_conditioned_gate", "low_rank_transport",
                "reset_classifier"):
        if method[key] is not False:
            raise ValueError(f"FO-Proto-MAML requires {key}=false")
    if config["test_checkpoint"] != "max-va.pth":
        raise ValueError("Test must use max-va.pth")


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
        self.optimizer = torch.optim.Adam(self.weights, lr=self.params["outer_lr"])
        self.best_score = -float("inf")
        self.best_state = None

    def meta_fit(self, meta_train_generator, meta_valid_generator):
        exp = self.config["experiment_config"]
        buffer = [torch.zeros_like(w) for w in self.weights]
        for i, task in enumerate(meta_train_generator(exp["train_iterations"])):
            self.meta_learner.train()
            support, labels, _ = task.support_set
            query, targets, _ = task.query_set
            fast = adapt(self.meta_learner, self.weights, support.to(self.dev),
                         labels.to(self.dev), self.params, task.num_ways)
            out, loss = query_loss(self.meta_learner, fast, query.to(self.dev),
                                   targets.to(self.dev))
            loss.backward()
            # Preserve FOMAML's per-task clipping and SUM over the meta-batch.
            for j, w in enumerate(self.weights):
                grad = w.grad
                if self.params["grad_clip"] is not None:
                    grad = grad.clamp(-self.params["grad_clip"], self.params["grad_clip"])
                buffer[j].add_(grad)
            self.optimizer.zero_grad(set_to_none=True)
            if (i + 1) % self.params["meta_batch_size"] == 0:
                for w, grad in zip(self.weights, buffer):
                    w.grad = grad
                self.optimizer.step()
                buffer = [torch.zeros_like(w) for w in self.weights]
                self.optimizer.zero_grad(set_to_none=True)
            self.log(task, out.detach().cpu().numpy(), loss.item())
            if (i + 1) % exp["validate_every"] == 0:
                self.meta_valid(meta_valid_generator)
        # Never silently label an unvalidated last snapshot as best-validation.
        if self.best_state is None:
            self.meta_valid(meta_valid_generator)
        return MyLearner(self.model_args, self.best_state, self.config, self.best_score)

    @torch.no_grad()
    def meta_valid(self, generator):
        self.meta_learner.eval()
        correct = total = 0
        for task in generator(self.config["experiment_config"]["validation_tasks"]):
            support, labels, _ = task.support_set
            query, targets, _ = task.query_set
            fast = adapt(self.meta_learner, self.weights, support.to(self.dev),
                         labels.to(self.dev), self.params, task.num_ways)
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
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in self.meta_learner.state_dict().items()}


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
        self.learner.load_state_dict(self.state)
        self.learner.eval()

    def fit(self, support_set):
        support, labels, _, ways, _ = support_set
        fast = adapt(self.learner, list(self.learner.parameters()),
                     support.to(self.dev), labels.to(self.dev),
                     self.config["method_config"], ways)
        return MyPredictor(self.learner, fast, self.dev)

    def save(self, path_to_save):
        path = Path(path_to_save)
        if not path.is_dir():
            raise ValueError("Checkpoint directory must exist")
        torch.save(dict(format_version=1, method="fo-proto-maml",
                        config=self.config, model_args=dict(self.model_args, dev="cpu"),
                        state=self.state, best_validation_accuracy=self.best_score),
                   path / "max-va.pth")

    def load(self, path_to_load):
        path = Path(path_to_load)
        checkpoint = path / "max-va.pth" if path.is_dir() else path
        if checkpoint.name != "max-va.pth":
            raise ValueError("Expected best-validation checkpoint max-va.pth")
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data["method"] != "fo-proto-maml" or data["format_version"] != 1:
            raise ValueError("Unsupported FO-Proto-MAML checkpoint")
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
