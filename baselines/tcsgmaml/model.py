""" Task-Conditioned Scalar-Gated MAML: an independent copy of SGMAML.

SGMAML gives every fast-weight tensor one static sigmoid gate that is shared by
all tasks. TCSGMAML keeps that shared logit a_j and adds a per-episode
correction produced from the support set alone:

    z_i     = (1 / |S_i|) * sum over x in S_i of f_frozen(x)
    delta_i = h_psi(z_i)
    m_ij    = sigmoid(a_j + delta_ij)
    theta'_ij = theta_ij - alpha * m_ij * g_ij

The frozen encoder f_frozen is loaded from an explicit checkpoint, never
trained and never part of the outer optimizer. The GateNet h_psi and the shared
logits are meta-parameters: they are excluded from the inner loop and learned
only through the query loss of meta-training tasks. Because the shared logits
start at 4.0 and the GateNet output layer starts at zero, the first update of a
TCSGMAML run is identical to the SGMAML update for every task.
"""
import os
import json
import random
import pickle
import contextlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Any, Dict, Tuple, List

from network import ResNet
from helpers_tcsgmaml import *
from task_conditioning import (FrozenTaskEncoder, GateNet, format_gate_summary,
                               rng_snapshot, summarize_gates)

from api import MetaLearner, Learner, Predictor

# --------------- MANDATORY ---------------
SEED = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# -----------------------------------------

# The ingestion program rebuilds the learner once per meta-test task, so the
# encoder package would otherwise be deserialized and rebuilt hundreds of
# times. Both caches hold frozen, read-only objects and are keyed so that a
# different checkpoint is never served from them.
_ENCODER_PACKAGE_CACHE = {}
_ENCODER_CACHE = {}


def load_config() -> dict:
    """ Read the configuration file that sits next to this module.

    Returns:
        dict: Parsed configuration.
    """
    config_path = Path(__file__).resolve().with_name("config.json")
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def read_encoder_package(path: str) -> dict:
    """ Read a stored encoder package, reusing the previous read when possible.

    Args:
        path (str): Path to the package file.

    Returns:
        dict: Encoder package.
    """
    stats = os.stat(path)
    key = (os.path.abspath(path), stats.st_mtime_ns, stats.st_size)
    package = _ENCODER_PACKAGE_CACHE.get(key)
    if package is None:
        with open(path, "rb") as package_file:
            package = pickle.load(package_file)
        _ENCODER_PACKAGE_CACHE.clear()
        _ENCODER_PACKAGE_CACHE[key] = package
    return package


def build_encoder(package: dict, dev: torch.device) -> FrozenTaskEncoder:
    """ Rebuild a frozen encoder, reusing the previous one when possible.

    Args:
        package (dict): Encoder package.
        dev (torch.device): Device where the data is located.

    Returns:
        FrozenTaskEncoder: Frozen encoder described by the package.
    """
    fingerprint = package.get("signature", {}).get("fingerprint")
    key = (fingerprint, str(dev), package.get("forward"))
    encoder = _ENCODER_CACHE.get(key) if fingerprint is not None else None
    if encoder is None:
        encoder = FrozenTaskEncoder.from_package(package, dev)
        if fingerprint is not None:
            _ENCODER_CACHE.clear()
            _ENCODER_CACHE[key] = encoder
    return encoder


class MyMetaLearner(MetaLearner):

    def __init__(self,
                 train_classes: int,
                 total_classes: int,
                 logger: Any) -> None:
        """ Defines the meta-learning algorithm's parameters. For example, one
        has to define what would be the meta-learner's architecture.

        Args:
            train_classes (int): Total number of classes that can be seen
                during meta-training. If the data format during training is
                'task', then this parameter corresponds to the number of ways,
                while if the data format is 'batch', this parameter corresponds
                to the total number of classes across all training datasets.
            total_classes (int): Total number of classes across all training
                datasets. If the data format during training is 'batch' this
                parameter is exactly the same as train_classes.
            logger (Logger): Logger that you can use during meta-learning
                (HIGHLY RECOMMENDED). You can use it after each meta-train or
                meta-validation iteration as follows:
                    self.log(data, predictions, loss, meta_train)
                - data (task or batch): It is the data used in the current
                    iteration.
                - predictions (np.ndarray): Predictions associated to each test
                    example in the specified data. It can be the raw logits
                    matrix (the logits are the unnormalized final scores of
                    your model), a probability matrix, or the predicted labels.
                - loss (float, optional): Loss of the current iteration.
                    Defaults to None.
                - meta_train (bool, optional): Boolean flag to control if the
                    current iteration belongs to meta-training. Defaults to
                    True.
        """
        # Note: the super().__init__() will set the following attributes:
        # - self.train_classes (int)
        # - self.total_classes (int)
        # - self.log (function) See the above description for details
        super().__init__(train_classes, total_classes, logger)

        # General data parameters
        config = load_config()
        experiment_config = config["experiment_config"]
        self.gate_init_logit = float(config.get("gate_init_logit", 4.0))
        encoder_config = dict(config.get("task_encoder", {}))
        gate_net_config = dict(config.get("gate_net", {}))
        self.gate_delta_scale = float(gate_net_config.get("delta_scale", 1.0))

        self.should_train = True
        self.ncc = False
        self.train_tasks = int(experiment_config["train_iterations"])
        self.val_tasks = int(experiment_config["validation_tasks"])
        self.val_after = int(experiment_config["validate_every"])

        # MAML parameters
        self.base_lr = 0.01
        self.grad_clip = 10
        self.second_order = False
        self.meta_batch_size = 2
        self.T = 5

        # General model parameters
        self.dev = self.get_device()
        self.opt_fn = torch.optim.Adam
        self.model_args = {
            "num_classes": self.train_classes,
            "dev": self.dev,
            "num_blocks": 18,
            "pretrained": False
        }

        # Meta-learner
        self.lr = 0.001
        self.meta_learner = ResNet(**self.model_args).to(self.dev)
        self.weights = [p.clone().detach().to(self.dev) for p in
            self.meta_learner.parameters()]
        for p in self.weights:
            p.requires_grad = True
        # Match the exact fast-weight order, including output weight and bias.
        # Scalar shape makes the final two gates independent of the task ways.
        self.gate_logits = nn.ParameterList([
            nn.Parameter(p.new_tensor(self.gate_init_logit))
            for p in self.weights
        ])
        self.num_gates = len(self.weights)

        # Frozen encoder and GateNet. Building them consumes the global torch
        # RNG, which the framework also uses to initialize the output layer of
        # every validation and test task, so the stream is restored afterwards
        # and stays identical to an SGMAML run with the same seed.
        with rng_snapshot():
            self.task_encoder = FrozenTaskEncoder.from_checkpoint(
                checkpoint=encoder_config.get("checkpoint"),
                dev=self.dev,
                num_blocks=int(encoder_config.get("num_blocks", 18)),
                img_size=int(encoder_config.get("img_size", 128)),
                forward_mode=str(encoder_config.get("forward",
                                                    "fast_weights")))
            self.gate_net = GateNet(
                in_features=self.task_encoder.out_features,
                hidden_size=int(gate_net_config.get("hidden_size", 128)),
                num_gates=self.num_gates,
                input_norm=str(gate_net_config.get("input_norm", "none"))
            ).to(self.dev)
        print(f"[tcsgmaml] frozen task encoder: "
              f"{json.dumps(self.task_encoder.signature)}")

        # The outer optimizer learns the initialization, the shared gate logits
        # and the GateNet together. The frozen encoder is never included.
        self.meta_parameters = (self.weights + list(self.gate_logits)
                                + list(self.gate_net.parameters()))
        self.optimizer = self.opt_fn(self.meta_parameters, lr=self.lr)

        # Store gradients across tasks
        self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in
            self.meta_parameters]

        # Validation-learner
        self.best_score = -float("inf")
        self.best_state = None
        self.best_gate_logits = None
        self.best_gate_net_state = None
        self.best_gate_summary = None
        self.gate_summaries = []
        self.completed_iterations = 0
        self.val_learner = ResNet(**self.model_args).to(self.dev)

    def meta_fit(self,
                 meta_train_generator: Iterable[Any],
                 meta_valid_generator: Iterable[Any]) -> Learner:
        """ Uses the generators to tune the meta-learner's parameters. The
        meta-training generator generates either few-shot learning tasks or
        batches of images, while the meta-valid generator always generates
        few-shot learning tasks.

        Args:
            meta_train_generator (Iterable[Any]): Function that generates the
                training data. The generated can be a N-way k-shot task or a
                batch of images with labels.
            meta_valid_generator (Iterable[Task]): Function that generates the
                validation data. The generated data always come in form of
                N-way k-shot tasks.

        Returns:
            Learner: Resulting learner ready to be trained and evaluated on new
                unseen tasks.
        """
        if self.should_train:
            self.optimizer.zero_grad()
            for i, task in enumerate(meta_train_generator(self.train_tasks)):
                self.meta_learner.train()
                self.gate_net.train()

                # Prepare data
                num_ways = task.num_ways
                X_train, y_train, _ = task.support_set
                X_train, y_train = X_train.to(self.dev), y_train.to(self.dev)
                X_test, y_test, _ = task.query_set
                X_test, y_test = X_test.to(self.dev), y_test.to(self.dev)

                # Describe the task and condition the gates on it. The
                # embedding is detached, but the GateNet and the sigmoid stay
                # in the graph so that the query loss trains both the GateNet
                # and the shared logits.
                gate_logits = self.task_gate_logits(X_train)

                # Compute loss
                task_weights = [p.clone() for p in self.weights]
                out, loss = self.compute_out_and_loss(self.meta_learner,
                    task_weights, gate_logits, X_train, y_train, X_test,
                    y_test, num_ways, True)

                # Propagate loss
                loss.backward()

                # Clip gradients
                if self.grad_clip is not None:
                    for p in self.meta_parameters:
                        if p.grad is not None:
                            p.grad = torch.clamp(p.grad, -self.grad_clip,
                                +self.grad_clip)

                # Update gradient buffer
                self.grad_buffer = [self.grad_buffer[j] + self.meta_parameters[j].grad
                    if self.meta_parameters[j].grad is not None else
                    self.grad_buffer[j] for j in range(len(self.meta_parameters))]
                self.optimizer.zero_grad()

                # Optimize metalearner
                if (i + 1) % self.meta_batch_size == 0:
                    for j, p in enumerate(self.meta_parameters):
                        p.grad = self.grad_buffer[j]
                    self.optimizer.step()

                    self.grad_buffer = [torch.zeros(p.size(), device=self.dev)
                        for p in self.meta_parameters]
                    self.optimizer.zero_grad()

                # Log iteration
                self.log(task, out.detach().cpu().numpy(), loss.item())

                self.completed_iterations = i + 1
                if (i + 1) % self.val_after == 0:
                    self.meta_valid(meta_valid_generator)

        if self.best_state is None:
            self.best_state = [p.clone().detach() for p in self.weights]
            self.best_gate_logits = [p.clone().detach() for p in self.gate_logits]
            self.best_gate_net_state = self.gate_net_state()

        maml_params = {
            "lr": self.base_lr,
            "grad_clip": self.grad_clip,
            "second_order": self.second_order,
            "T": self.T,
            "ncc": self.ncc,
            "gate_delta_scale": self.gate_delta_scale
        }
        gate_net_package = {
            "config": self.gate_net.config(),
            "state_dict": self.best_gate_net_state,
            "delta_scale": self.gate_delta_scale
        }
        return MyLearner(self.model_args, self.meta_learner.state_dict(),
            self.best_state, maml_params, self.best_gate_logits,
            gate_net_package, self.task_encoder.package(),
            {"validations": self.gate_summaries,
             "best": self.best_gate_summary})

    def gate_net_state(self) -> Dict[str, torch.Tensor]:
        """ Take a detached copy of the current GateNet parameters.

        Returns:
            Dict[str, torch.Tensor]: GateNet state dict on the CPU.
        """
        return {name: value.detach().cpu().clone() for name, value in
            self.gate_net.state_dict().items()}

    def task_gate_logits(self,
                         X_train: torch.Tensor,
                         detach: bool = False) -> List[torch.Tensor]:
        """ Produce the gate logits of one episode from its support set.

        The embedding is computed once per episode, before any adaptation, and
        the resulting logits are reused by every inner step of that episode.

        Args:
            X_train (torch.Tensor): Support set images.
            detach (bool, optional): If True, no graph is kept. Used whenever
                no meta-gradient will be taken. Defaults to False.

        Returns:
            List[torch.Tensor]: One scalar logit per fast-weight tensor.
        """
        embedding = self.task_encoder.embed(X_train)
        if detach:
            with torch.no_grad():
                deltas = self.gate_net(embedding)
            shared = [logit.detach() for logit in self.gate_logits]
        else:
            deltas = self.gate_net(embedding)
            shared = list(self.gate_logits)
        if deltas.numel() != self.num_gates:
            raise ValueError(
                f"The GateNet produced {deltas.numel()} corrections for "
                f"{self.num_gates} fast-weight tensors.")
        if self.gate_delta_scale != 1.0:
            deltas = deltas * self.gate_delta_scale
        return [shared[j] + deltas[j] for j in range(self.num_gates)]

    def meta_valid(self, meta_valid_generator: Iterable[Any]) -> None:
        """ Evaluate the current meta-learner with the meta-validation split
        to select the best model.

        Args:
            meta_valid_generator (Iterable[Task]): Function that generates the
                validation data. The generated data always come in form of
                N-way k-shot tasks.
        """
        total_test_images = 0
        correct_predictions = 0
        realized_gate_logits = []
        self.gate_net.eval()
        for task in meta_valid_generator(self.val_tasks):
            # Prepare data
            num_ways = task.num_ways
            X_train, y_train, _ = task.support_set
            X_train, y_train = X_train.to(self.dev), y_train.to(self.dev)
            X_test, y_test, _ = task.query_set
            X_test = X_test.to(self.dev)

            # Adapt learner
            self.val_learner.load_params(self.meta_learner.state_dict())
            self.val_learner.eval()
            self.val_learner.modify_out_layer(num_ways)

            # Prepare weights
            task_weights = [p.clone() for p in self.weights[:-2]]
            val_weights = [p.clone().detach().to(self.dev) for p in
                self.val_learner.parameters()]
            task_weights.extend(val_weights[-2:])
            for p in task_weights[-2:]:
                p.requires_grad = True

            # Condition the gates on this validation task. Nothing is learned
            # here, so no graph is needed.
            gate_logits = self.task_gate_logits(X_train, detach=True)
            realized_gate_logits.append(
                torch.stack(gate_logits).detach().cpu().numpy())

            # Evaluate learner
            out, _ = self.compute_out_and_loss(self.val_learner, task_weights,
                gate_logits, X_train, y_train, X_test, y_test, num_ways, False,
                True)
            preds = torch.argmax(out, dim=1).cpu().numpy()

            # Log iteration
            self.log(task, out.cpu().numpy(), meta_train=False)

            # Keep track of scores
            total_test_images += len(y_test)
            correct_predictions += np.sum(preds == y_test.numpy())

        # Summarize how much the gates actually moved between tasks. A run
        # whose across-task spread stays at zero never used its conditioning.
        summary = None
        if realized_gate_logits:
            summary = summarize_gates(np.stack(realized_gate_logits))
            summary["iteration"] = self.completed_iterations
            self.gate_summaries.append(summary)
            print(format_gate_summary(summary, self.completed_iterations))

        # Check if the accuracy is better and store the new best state
        val_acc = correct_predictions / total_test_images
        if val_acc > self.best_score:
            self.best_score = val_acc
            self.best_state = [p.clone().detach() for p in self.weights]
            self.best_gate_logits = [p.clone().detach() for p in self.gate_logits]
            self.best_gate_net_state = self.gate_net_state()
            self.best_gate_summary = summary

    def get_device(self) -> torch.device:
        """ Get the current device, it can be CPU or GPU.

        Returns:
            torch.device: Available device.
        """
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    def compute_out_and_loss(self,
                             model: nn.Module,
                             weights: List[torch.Tensor],
                             gate_logits: List[torch.Tensor],
                             X_train: torch.Tensor,
                             y_train: torch.Tensor,
                             X_test: torch.Tensor,
                             y_test: torch.Tensor,
                             num_classes: int,
                             training: bool,
                             no_loss: bool = False) -> Tuple[torch.Tensor,
                                                             torch.Tensor]:
        """ Compute the output and loss using the specified data.

        Args:
            model (nn.Module): Model to be used.
            weights (List[torch.Tensor]): Weights to be used by the model.
            gate_logits (List[torch.Tensor]): Gate logits of the current
                episode, one scalar per fast-weight tensor.
            X_train (torch.Tensor): Support set images.
            y_train (torch.Tensor): Support set labels.
            X_test (torch.Tensor): Query set images.
            y_test (torch.Tensor): Query set labels.
            num_classes (int): Number of classes to predict.
            training (bool): Boolean flag to control the execution context. If
                True, keep track of the gradients, otherwise the gradients are
                ignored.
            no_loss (bool, optional): Boolean flag to control the loss
                computation. If True, the loss is not computed, otherwise the
                loss is computed. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output and loss.
        """
        # Inner step
        perform_innet_step = False if self.ncc and not training else True
        if perform_innet_step:
            retain_graph = self.second_order or self.T > 1
            for _ in range(self.T):
                # Compute gradients
                if self.ncc:
                    grads = get_grads_ncc(model, X_train, y_train, X_test,
                        y_test, num_classes, weights, self.second_order,
                        retain_graph)
                else:
                    grads = get_grads(model, X_train, y_train, weights,
                        self.second_order, retain_graph)

                # Update task weights
                weights = update_weights(weights, grads, self.grad_clip,
                    self.base_lr, gate_logits)

        # Use torch.no_grad when evaluating
        if training:
            context = self.empty_context
        else:
            context = torch.no_grad

        # Get and return performance on query set
        with context():
            if self.ncc:
                prototypes = process_support_set(model, weights, X_train,
                    y_train, num_classes)
                out = process_query_set(model, weights, X_test, prototypes)
            else:
                out = model.forward_weights(X_test, weights)

            if no_loss:
                loss = None
            else:
                reg = num_classes * len(y_test) if self.ncc else 1
                loss = model.criterion(out, y_test) / reg

        return out, loss

    @contextlib.contextmanager
    def empty_context(self) -> None:
        """ Defines an empty context to avoid computing unnecessary gradients.
        """
        yield None


class MyLearner(Learner):

    def __init__(self,
                 model_args: dict = {},
                 model_state: dict = {},
                 weights: List[torch.Tensor] = [],
                 maml_params: dict = {},
                 gate_logits: List[torch.Tensor] = None,
                 gate_net_package: dict = None,
                 task_encoder_package: dict = None,
                 gate_stats: dict = None) -> None:
        """ Defines the learner initialization.

        Args:
            model_args (dict, optional): Arguments to initialize the learner.
                Defaults to {}.
            model_state (dict, optional): Weights to initialize the learner.
                Defaults to {}.
            weights (List[torch.Tensor], optional): Best weights found by the
                meta-learner. Defaults to [].
            maml_params (dict, optional): Parameters required by MAML. Defaults
                to {}.
            gate_logits (List[torch.Tensor], optional): Shared gate logits from
                the same validation checkpoint as weights. Defaults to None.
            gate_net_package (dict, optional): GateNet architecture and weights
                from that same checkpoint. Defaults to None.
            task_encoder_package (dict, optional): Everything needed to rebuild
                the frozen encoder without the original file. Defaults to None.
            gate_stats (dict, optional): Gate diagnostics recorded during
                meta-training. Defaults to None.
        """
        super().__init__()
        self.model_args = model_args
        self.model_state = model_state
        self.weights = weights
        self.maml_params = maml_params
        self.gate_logits = ([] if gate_logits is None else
                            [p.clone().detach() for p in gate_logits])
        self.gate_net_package = gate_net_package
        self.task_encoder_package = task_encoder_package
        self.gate_stats = {} if gate_stats is None else gate_stats
        self.gate_net = None
        self.task_encoder = None
        self.gate_delta_scale = float(
            (maml_params or {}).get("gate_delta_scale", 1.0))

    def task_gate_logits(self, X_train: torch.Tensor) -> List[torch.Tensor]:
        """ Produce the gate logits of a new unseen task.

        Nothing is learned at meta-test time: the frozen encoder, the GateNet
        and the shared logits are all fixed, and only the fast weights of the
        main model move.

        Args:
            X_train (torch.Tensor): Support set images.

        Returns:
            List[torch.Tensor]: One scalar logit per fast-weight tensor.
        """
        embedding = self.task_encoder.embed(X_train)
        with torch.no_grad():
            deltas = self.gate_net(embedding)
            if self.gate_delta_scale != 1.0:
                deltas = deltas * self.gate_delta_scale
        if deltas.numel() != len(self.gate_logits):
            raise ValueError(
                f"The GateNet produced {deltas.numel()} corrections for "
                f"{len(self.gate_logits)} fast-weight tensors.")
        return [self.gate_logits[j] + deltas[j]
                for j in range(len(self.gate_logits))]

    def fit(self, support_set: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                               int, int]) -> Predictor:
        """ Fit the Learner to the support set of a new unseen task.

        Args:
            support_set (Tuple[Tensor, Tensor, Tensor, int, int]): Support set
                of a task. The data arrive in the following format (X_train,
                y_train, original_y_train, n_ways, k_shots). X_train is the
                tensor of labeled images of shape [n_ways*k_shots x 3 x 128 x
                128], y_train is the tensor of encoded labels (Long) for each
                image in X_train with shape of [n_ways*k_shots],
                original_y_train is the tensor of original labels (Long) for
                each image in X_train with shape of [n_ways*k_shots], n_ways is
                the number of classes and k_shots the number of examples per
                class.

        Returns:
            Predictor: The resulting predictor ready to predict unlabelled
                query image examples from new unseen tasks.
        """
        X_train, y_train, _, n_ways, _ = support_set
        X_train, y_train = X_train.to(self.dev), y_train.to(self.dev)

        # Adapt learner
        self.learner.modify_out_layer(n_ways)

        # Prepare weights
        task_weights = [p.clone() for p in self.weights[:-2]]
        learner_weights = [p.clone().detach().to(self.dev) for p in
            self.learner.parameters()]
        task_weights.extend(learner_weights[-2:])
        for p in task_weights[-2:]:
            p.requires_grad = True

        # A fresh task embedding and a fresh set of gates for every support
        # set. The scalar gates stay valid when the head changes shape.
        gate_logits = self.task_gate_logits(X_train)

        if self.ncc:
            with torch.no_grad():
                prototypes = process_support_set(self.learner, task_weights,
                    X_train, y_train, n_ways)

        else:
            prototypes = None
            # Fit weights
            retain_graph = self.second_order or self.T > 1
            for _ in range(self.T):
                grads = get_grads(self.learner, X_train, y_train, task_weights,
                    self.second_order, retain_graph)
                task_weights = update_weights(task_weights, grads,
                    self.grad_clip, self.lr, gate_logits)

        return MyPredictor(self.learner, task_weights, self.dev, prototypes)

    def save(self, path_to_save: str) -> None:
        """ Saves the learning object associated to the Learner.

        Args:
            path_to_save (str): Path where the learning object will be saved.
        """

        if not os.path.isdir(path_to_save):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))

        with open(f"{path_to_save}/model_args.pickle", "wb+") as f:
            pickle.dump(self.model_args, f)
        with open(f"{path_to_save}/model_state.pickle", "wb+") as f:
            pickle.dump(self.model_state, f)
        with open(f"{path_to_save}/weights.pickle", "wb+") as f:
            pickle.dump(self.weights, f)
        with open(f"{path_to_save}/maml_params.pickle", "wb+") as f:
            pickle.dump(self.maml_params, f)
        with open(f"{path_to_save}/gate_logits.pickle", "wb+") as f:
            pickle.dump(self.gate_logits, f)
        with open(f"{path_to_save}/gate_net.pickle", "wb+") as f:
            pickle.dump(self.gate_net_package, f)
        with open(f"{path_to_save}/task_encoder.pickle", "wb+") as f:
            pickle.dump(self.task_encoder_package, f)
        with open(f"{path_to_save}/gate_stats.pickle", "wb+") as f:
            pickle.dump(self.gate_stats, f)

    def load(self, path_to_load: str) -> None:
        """ Loads the learning object associated to the Learner. It should
        match the way you saved this object in self.save().

        Args:
            path_to_load (str): Path where the Learner is saved.
        """
        if not os.path.isdir(path_to_load):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))

        model_args_file = f"{path_to_load}/model_args.pickle"
        if os.path.isfile(model_args_file):
            with open(model_args_file, "rb") as f:
                self.model_args = pickle.load(f)
            self.dev = self.model_args["dev"]
            self.learner = ResNet(**self.model_args).to(self.dev)
        else:
            raise Exception(f"'{model_args_file}' not found")

        model_state_file = f"{path_to_load}/model_state.pickle"
        if os.path.isfile(model_state_file):
            with open(model_state_file, "rb") as f:
                state = pickle.load(f)
            self.learner.load_params(state)
            self.learner.eval()
        else:
            raise Exception(f"'{model_state_file}' not found")

        weights_file = f"{path_to_load}/weights.pickle"
        if os.path.isfile(weights_file):
            with open(weights_file, "rb") as f:
                self.weights = pickle.load(f)
            for p in self.weights:
                p.requires_grad = True
        else:
            raise Exception(f"'{weights_file}' not found")

        gate_logits_file = f"{path_to_load}/gate_logits.pickle"
        if os.path.isfile(gate_logits_file):
            with open(gate_logits_file, "rb") as f:
                gate_logits = pickle.load(f)
            if len(gate_logits) != len(self.weights):
                raise ValueError("Checkpoint must contain one gate per fast-weight tensor.")
            if any(not isinstance(p, torch.Tensor) or p.ndim != 0
                   for p in gate_logits):
                raise ValueError("Checkpoint gate logits must be scalar tensors.")
            # Learned gates are fixed during meta-test; never reset to 4.0.
            self.gate_logits = [p.detach().to(self.dev) for p in gate_logits]
        else:
            raise Exception(f"'{gate_logits_file}' not found")

        # The frozen encoder travels with the learner, so meta-test never
        # depends on the original checkpoint path still being reachable.
        task_encoder_file = f"{path_to_load}/task_encoder.pickle"
        if os.path.isfile(task_encoder_file):
            self.task_encoder_package = read_encoder_package(task_encoder_file)
            if not isinstance(self.task_encoder_package, dict):
                raise ValueError("The stored task encoder is not a package.")
            self.task_encoder = build_encoder(self.task_encoder_package,
                                              self.dev)
        else:
            raise Exception(f"'{task_encoder_file}' not found")

        gate_net_file = f"{path_to_load}/gate_net.pickle"
        if os.path.isfile(gate_net_file):
            with open(gate_net_file, "rb") as f:
                self.gate_net_package = pickle.load(f)
            if (not isinstance(self.gate_net_package, dict)
                    or "config" not in self.gate_net_package
                    or self.gate_net_package.get("state_dict") is None):
                raise ValueError(
                    "The stored GateNet package must contain both its "
                    "architecture and its weights. A checkpoint without them "
                    "is never replaced by a randomly initialized GateNet.")
            gate_net_config = dict(self.gate_net_package["config"])
            if gate_net_config["num_gates"] != len(self.weights):
                raise ValueError(
                    f"The stored GateNet produces "
                    f"{gate_net_config['num_gates']} gates for "
                    f"{len(self.weights)} fast-weight tensors.")
            if gate_net_config["in_features"] != self.task_encoder.out_features:
                raise ValueError(
                    f"The stored GateNet expects "
                    f"{gate_net_config['in_features']} embedding features "
                    f"while the encoder produces "
                    f"{self.task_encoder.out_features}.")
            with rng_snapshot():
                self.gate_net = GateNet(**gate_net_config).to(self.dev)
            self.gate_net.load_state_dict(
                {name: value.to(self.dev) for name, value in
                 self.gate_net_package["state_dict"].items()})
            self.gate_net.eval()
            for parameter in self.gate_net.parameters():
                parameter.requires_grad_(False)
            self.gate_delta_scale = float(
                self.gate_net_package.get("delta_scale", 1.0))
        else:
            raise Exception(f"'{gate_net_file}' not found")

        gate_stats_file = f"{path_to_load}/gate_stats.pickle"
        if os.path.isfile(gate_stats_file):
            with open(gate_stats_file, "rb") as f:
                self.gate_stats = pickle.load(f)

        maml_params_file = f"{path_to_load}/maml_params.pickle"
        if os.path.isfile(maml_params_file):
            with open(maml_params_file, "rb") as f:
                self.maml_params = pickle.load(f)
            self.lr = self.maml_params["lr"]
            self.grad_clip = self.maml_params["grad_clip"]
            self.second_order = self.maml_params["second_order"]
            self.T = self.maml_params["T"]
            self.ncc = self.maml_params["ncc"]
            self.gate_delta_scale = float(
                self.maml_params.get("gate_delta_scale",
                                     self.gate_delta_scale))
        else:
            raise Exception(f"'{maml_params_file}' not found")


class MyPredictor(Predictor):

    def __init__(self,
                 model: nn.Module,
                 weights: List[torch.Tensor],
                 dev: torch.device,
                 prototypes: torch.Tensor) -> None:
        """Defines the Predictor initialization.

        Args:
            model (nn.Module): Fitted learner.
            weights (List[torch.Tensor]): Best weights for the model.
            dev (torch.device): Device where the data is located.
            prototypes (torch.Tensor): Support prototypes.
        """
        super().__init__()
        self.model = model
        self.weights = weights
        self.dev = dev
        self.prototypes = prototypes

    def predict(self, query_set: torch.Tensor) -> np.ndarray:
        """ Given a query_set, predicts the probabilities associated to the
        provided images or the labels to the provided images.

        Args:
            query_set (Tensor): Tensor of unlabelled image examples of shape
                [n_ways*query_size x 3 x 128 x 128].

        Returns:
            np.ndarray: It can be:
                - Raw logits matrix (the logits are the unnormalized final
                    scores of your model). The matrix must be of shape
                    [n_ways*query_size, n_ways].
                - Predicted label probabilities matrix. The matrix must be of
                    shape [n_ways*query_size, n_ways].
                - Predicted labels. The array must be of shape
                    [n_ways*query_size].
        """
        X_test = query_set.to(self.dev)
        with torch.no_grad():
            if self.prototypes is not None:
                out = process_query_set(self.model, self.weights, X_test,
                    self.prototypes)
            else:
                out = self.model.forward_weights(X_test, self.weights)
            probs = F.softmax(out, dim=1).cpu().numpy()

        return probs
