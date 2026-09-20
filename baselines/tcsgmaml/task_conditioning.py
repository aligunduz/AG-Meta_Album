""" Task conditioning components used only by the TCSGMAML baseline.

Three independent pieces live here, all of them local to this submission:

* ``FrozenTaskEncoder``  - a feature extractor that never learns. It turns the
  support images of one episode into a single task embedding.
* ``GateNet``            - a small MLP that maps that embedding to one scalar
  gate correction per adapted parameter tensor.
* gate bookkeeping       - helpers that summarize the realized gates so that a
  run can be diagnosed without changing the shared logger API.

Nothing in this file touches the main model, the shared framework code or any
other baseline.
"""
import hashlib
import os
import pickle
import contextlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import ResNet


PACKAGE_FORMAT = 1
FORWARD_MODES = ("fast_weights", "module_eval")
INPUT_NORMS = ("none", "l2", "layernorm")


@contextlib.contextmanager
def rng_snapshot() -> None:
    """ Run a block without leaving any trace in the global torch RNG.

    The framework draws from the global torch RNG every time a task resizes the
    output layer (``ResNet.modify_out_layer``), both during meta-validation and
    during meta-testing. Building the frozen encoder and the GateNet would
    otherwise shift that stream and make TCSGMAML and SGMAML runs that share a
    seed sample different output-layer initializations, which would break the
    paired comparison between the two baselines. Task sampling is unaffected
    either way, since the data generator owns a separate numpy RandomState.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else None)
    try:
        yield None
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _load_any(path: str) -> Any:
    """ Load a checkpoint saved either with pickle or with torch.save.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        Any: Deserialized object.
    """
    if os.path.splitext(path)[1].lower() in (".pickle", ".pkl"):
        with open(path, "rb") as checkpoint_file:
            return pickle.load(checkpoint_file)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Research checkpoints written by older torch versions are plain
        # pickles of tensor containers, which weights_only rejects.
        return torch.load(path, map_location="cpu")


def read_encoder_checkpoint(path: str) -> Tuple[Optional[List[torch.Tensor]],
                                                Optional[OrderedDict]]:
    """ Read an encoder checkpoint and report which form it has.

    Two forms are accepted, both of them produced by this framework:
        * a list of tensors in ``forward_weights`` order, which is what
          ``Learner.save`` writes to ``weights.pickle``;
        * a module ``state_dict``, which additionally carries BatchNorm
          buffers.
    A dict wrapping either form under 'weights', 'state_dict' or 'model_state'
    is unwrapped first.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        Tuple[Optional[List[torch.Tensor]], Optional[OrderedDict]]: The weight
            list and the state dict. Exactly one of them is not None.
    """
    checkpoint = _load_any(path)

    if isinstance(checkpoint, dict):
        for key in ("weights", "state_dict", "model_state"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    if isinstance(checkpoint, (list, tuple)):
        weights = list(checkpoint)
        if not weights or any(not torch.is_tensor(w) for w in weights):
            raise ValueError(
                f"'{path}' holds a list that is empty or contains non-tensor "
                "entries. A weight-list checkpoint must be the list of "
                "tensors written by Learner.save (weights.pickle).")
        return weights, None

    if isinstance(checkpoint, dict):
        if not checkpoint or any(not torch.is_tensor(v)
                                 for v in checkpoint.values()):
            raise ValueError(
                f"'{path}' holds a dict that is empty or contains non-tensor "
                "values, so it is not a usable state dict.")
        return None, OrderedDict(checkpoint)

    raise ValueError(
        f"'{path}' holds {type(checkpoint).__name__}, which is neither a list "
        "of tensors nor a state dict. Point 'task_encoder.checkpoint' at a "
        "weights.pickle file or at a torch-saved state dict.")


def _tensor_fingerprint(tensors: List[torch.Tensor]) -> str:
    """ Compute a stable fingerprint of the encoder parameters.

    Args:
        tensors (List[torch.Tensor]): Tensors to fingerprint.

    Returns:
        str: Hexadecimal digest.
    """
    digest = hashlib.sha1()
    for tensor in tensors:
        array = tensor.detach().to("cpu").contiguous().numpy()
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _bn_statistics_are_defaults(state: OrderedDict) -> bool:
    """ Check whether the BatchNorm buffers were ever updated.

    ``ResNet.forward_weights`` normalizes with batch statistics and never
    writes to the module buffers, so a checkpoint trained through that path
    keeps running_mean at zero and running_var at one. Using such a checkpoint
    in eval mode would silently skip normalization altogether.

    Args:
        state (OrderedDict): State dict to inspect.

    Returns:
        bool: True when every BatchNorm buffer still holds its initial value.
    """
    seen = False
    for key, value in state.items():
        if key.endswith("running_mean"):
            seen = True
            if not torch.allclose(value, torch.zeros_like(value)):
                return False
        elif key.endswith("running_var"):
            seen = True
            if not torch.allclose(value, torch.ones_like(value)):
                return False
    return seen


class FrozenTaskEncoder:
    """ Support-only task descriptor that is never trained.

    The encoder is completely independent of the adapted model: its parameters
    are excluded from the outer optimizer, it is kept in eval mode, features
    are extracted under ``torch.no_grad`` and the resulting embedding is
    detached before it reaches the GateNet.
    """

    def __init__(self,
                 network: ResNet,
                 weights: List[torch.Tensor],
                 forward_mode: str,
                 has_bn_statistics: bool,
                 signature: Dict[str, Any],
                 dev: torch.device) -> None:
        """ Store an already prepared encoder. Use the two class methods below.

        Args:
            network (ResNet): Architecture container, already loaded.
            weights (List[torch.Tensor]): Frozen weights in forward_weights
                order.
            forward_mode (str): Either 'fast_weights' or 'module_eval'.
            has_bn_statistics (bool): Whether real BatchNorm buffers are known.
            signature (Dict[str, Any]): Provenance record of the checkpoint.
            dev (torch.device): Device where the data is located.
        """
        self.network = network
        self.weights = weights
        self.forward_mode = forward_mode
        self.has_bn_statistics = has_bn_statistics
        self.signature = signature
        self.dev = dev
        self.out_features = int(network.in_features)

    @staticmethod
    def _build_network(num_blocks: int,
                       img_size: int,
                       dev: torch.device) -> ResNet:
        """ Build the architecture container used to extract features.

        Args:
            num_blocks (int): Number of ResNet blocks.
            img_size (int): Size of the images the encoder will process.
            dev (torch.device): Device where the data is located.

        Returns:
            ResNet: Network in eval mode with every parameter frozen.
        """
        network = ResNet(num_classes=2, dev=dev, pretrained=False,
                         num_blocks=num_blocks, img_size=img_size).to(dev)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        return network

    @staticmethod
    def _validate_against(network: ResNet,
                          weights: List[torch.Tensor],
                          source: str) -> None:
        """ Check that a weight list matches the encoder architecture.

        Args:
            network (ResNet): Architecture container.
            weights (List[torch.Tensor]): Candidate weights.
            source (str): Description used in error messages.

        Raises:
            ValueError: If the number of tensors or any shape disagrees.
        """
        reference = list(network.parameters())
        if len(weights) != len(reference):
            raise ValueError(
                f"{source} holds {len(weights)} tensors but the encoder "
                f"architecture has {len(reference)}. The checkpoint was "
                "probably produced with a different backbone.")
        # The last two tensors are the output layer, which the task embedding
        # never uses, so their number of ways is allowed to differ.
        for index, (candidate, expected) in enumerate(
                zip(weights[:-2], reference[:-2])):
            if candidate.shape != expected.shape:
                raise ValueError(
                    f"{source} has shape {tuple(candidate.shape)} at tensor "
                    f"{index} while the encoder expects "
                    f"{tuple(expected.shape)}.")

    @classmethod
    def from_checkpoint(cls,
                        checkpoint: Optional[str],
                        dev: torch.device,
                        num_blocks: int = 18,
                        img_size: int = 128,
                        forward_mode: str = "fast_weights"
                        ) -> "FrozenTaskEncoder":
        """ Build the encoder from an explicit checkpoint path.

        There is deliberately no fallback: a missing, unreadable or
        architecturally incompatible checkpoint raises instead of quietly
        producing a random or ImageNet-pretrained encoder.

        Args:
            checkpoint (Optional[str]): Path to the checkpoint file.
            dev (torch.device): Device where the data is located.
            num_blocks (int, optional): Number of ResNet blocks. Defaults to 18.
            img_size (int, optional): Image size. Defaults to 128.
            forward_mode (str, optional): Feature extraction path. Defaults to
                'fast_weights'.

        Returns:
            FrozenTaskEncoder: Ready to use encoder.
        """
        if forward_mode not in FORWARD_MODES:
            raise ValueError(
                f"'task_encoder.forward' must be one of {FORWARD_MODES}, "
                f"received '{forward_mode}'.")
        if checkpoint is None or not str(checkpoint).strip():
            raise ValueError(
                "TCSGMAML needs an explicit frozen task-encoder checkpoint. "
                "Set 'task_encoder.checkpoint' in baselines/tcsgmaml/"
                "config.json to a checkpoint trained on meta-train data only, "
                "for example <ingestion_output>/model/weights.pickle from an "
                "earlier run of this framework. The baseline never falls back "
                "to a random or ImageNet-pretrained encoder.")

        path = os.path.abspath(os.path.expanduser(str(checkpoint)))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Frozen task-encoder checkpoint '{path}' does not exist. "
                "'task_encoder.checkpoint' must point at an existing file.")

        weights, state = read_encoder_checkpoint(path)
        network = cls._build_network(num_blocks, img_size, dev)
        has_bn_statistics = False

        if state is not None:
            usable_state = OrderedDict(
                (key, value) for key, value in state.items()
                if not key.startswith("model.out."))
            has_bn_statistics = not _bn_statistics_are_defaults(usable_state)
            incompatible = network.load_state_dict(usable_state, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            if unexpected:
                raise ValueError(
                    f"'{path}' contains keys that do not belong to the encoder "
                    f"architecture, for example '{unexpected[0]}'.")
            loaded = [key for key in usable_state if "running" not in key
                      and "num_batches_tracked" not in key]
            if not loaded:
                raise ValueError(
                    f"'{path}' contains no encoder parameters, only buffers.")
            weights = [p.detach() for p in network.parameters()]
        else:
            cls._validate_against(network, weights, f"'{path}'")
            weights = [w.detach().to(dev) for w in weights]
            # Mirror the checkpoint into the module so that the saved package
            # and the weight list describe the same encoder.
            with torch.no_grad():
                for parameter, weight in zip(network.parameters(), weights):
                    if parameter.shape == weight.shape:
                        parameter.copy_(weight)

        if forward_mode == "module_eval" and not has_bn_statistics:
            raise ValueError(
                f"'{path}' carries no usable BatchNorm statistics, so "
                "'task_encoder.forward': 'module_eval' would extract features "
                "without normalization. Checkpoints trained through "
                "ResNet.forward_weights never update those buffers. Use "
                "'fast_weights', which normalizes with support-batch "
                "statistics exactly as the network was trained.")

        for weight in weights:
            weight.requires_grad_(False)

        signature = {
            "checkpoint": path,
            "forward": forward_mode,
            "num_blocks": num_blocks,
            "img_size": img_size,
            "num_tensors": len(weights),
            "num_parameters": int(sum(w.numel() for w in weights)),
            "has_bn_statistics": has_bn_statistics,
            "fingerprint": _tensor_fingerprint(weights),
        }
        return cls(network, weights, forward_mode, has_bn_statistics,
                   signature, dev)

    @classmethod
    def from_package(cls,
                     package: Dict[str, Any],
                     dev: torch.device) -> "FrozenTaskEncoder":
        """ Rebuild the encoder from the package stored in a checkpoint.

        Meta-testing must not depend on the original checkpoint path still
        being reachable, so the learner carries the encoder with it.

        Args:
            package (Dict[str, Any]): Package produced by ``self.package()``.
            dev (torch.device): Device where the data is located.

        Returns:
            FrozenTaskEncoder: Ready to use encoder.
        """
        required = ("format", "forward", "num_blocks", "img_size", "weights",
                    "buffers", "has_bn_statistics", "signature")
        missing = [key for key in required if key not in package]
        if missing:
            raise ValueError(
                "The stored task-encoder package is incomplete, missing "
                f"{missing}. Re-run meta-training with this baseline.")
        if package["format"] != PACKAGE_FORMAT:
            raise ValueError(
                f"Unsupported task-encoder package format "
                f"{package['format']}, expected {PACKAGE_FORMAT}.")

        with rng_snapshot():
            network = cls._build_network(int(package["num_blocks"]),
                                         int(package["img_size"]), dev)
        weights = [w.detach().to(dev) for w in package["weights"]]
        cls._validate_against(network, weights, "The stored encoder package")

        buffers = OrderedDict(
            (key, value.to(dev)) for key, value in package["buffers"].items())
        if buffers:
            network.load_state_dict(buffers, strict=False)
        with torch.no_grad():
            for parameter, weight in zip(network.parameters(), weights):
                if parameter.shape == weight.shape:
                    parameter.copy_(weight)
        for weight in weights:
            weight.requires_grad_(False)

        signature = dict(package["signature"])
        fingerprint = _tensor_fingerprint(weights)
        if signature.get("fingerprint") not in (None, fingerprint):
            raise ValueError(
                "The stored encoder weights do not match their recorded "
                "fingerprint, so the checkpoint is inconsistent.")
        signature["fingerprint"] = fingerprint
        return cls(network, weights, str(package["forward"]),
                   bool(package["has_bn_statistics"]), signature, dev)

    def package(self) -> Dict[str, Any]:
        """ Serialize everything needed to rebuild this encoder.

        Returns:
            Dict[str, Any]: Package to be stored next to the learner.
        """
        parameter_names = {name for name, _ in self.network.named_parameters()}
        buffers = OrderedDict(
            (key, value.detach().cpu())
            for key, value in self.network.state_dict().items()
            if key not in parameter_names)
        return {
            "format": PACKAGE_FORMAT,
            "forward": self.forward_mode,
            "num_blocks": int(self.network.num_blocks),
            "img_size": int(self.signature["img_size"]),
            "weights": [w.detach().cpu() for w in self.weights],
            "buffers": buffers,
            "has_bn_statistics": self.has_bn_statistics,
            "signature": dict(self.signature),
        }

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """ Describe one episode by the mean frozen feature of its support set.

            z_i = (1 / |S_i|) * sum over x in S_i of f_frozen(x)

        Only support images take part. Query images, query labels and the
        domain identity are never seen by the encoder.

        Note that in 'fast_weights' mode the features are normalized with the
        statistics of the support batch itself, which is how every network in
        this framework is trained. The whole support set is therefore passed in
        a single batch and is never chunked, because a different batch
        composition would produce a different embedding.

        Args:
            images (torch.Tensor): Support images of one episode, with shape
                [num_ways*num_shots x 3 x img_size x img_size].

        Returns:
            torch.Tensor: Detached task embedding of shape [out_features].
        """
        if images.dim() != 4 or images.size(0) < 1:
            raise ValueError(
                "The task embedding needs a non-empty batch of support images "
                f"with four dimensions, received shape {tuple(images.shape)}.")
        with torch.no_grad():
            batch = images.to(self.dev)
            if self.forward_mode == "module_eval":
                features = self.network.compute_in_features(batch)
            else:
                features = self.network.forward_weights(batch, self.weights,
                                                        embedding=True)
            embedding = features.mean(dim=0)
        return embedding.detach()


class GateNet(nn.Module):
    """ Map a task embedding to one scalar gate correction per tensor.

    The output layer starts at exactly zero, so the first episode of
    meta-training produces the same gates as SGMAML for every task. Any later
    deviation is something the query loss decided to learn.
    """

    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 num_gates: int,
                 input_norm: str = "none") -> None:
        """ Define the GateNet.

        Args:
            in_features (int): Size of the task embedding.
            hidden_size (int): Width of the hidden layer.
            num_gates (int): Number of adapted parameter tensors.
            input_norm (str, optional): Embedding normalization, one of 'none',
                'l2' or 'layernorm'. 'none' reproduces the plain
                Linear-ReLU-Linear design. Defaults to 'none'.
        """
        super().__init__()
        if input_norm not in INPUT_NORMS:
            raise ValueError(
                f"'gate_net.input_norm' must be one of {INPUT_NORMS}, "
                f"received '{input_norm}'.")
        self.in_features = int(in_features)
        self.hidden_size = int(hidden_size)
        self.num_gates = int(num_gates)
        self.input_norm = input_norm
        self.norm = (nn.LayerNorm(self.in_features)
                     if input_norm == "layernorm" else nn.Identity())
        self.hidden = nn.Linear(self.in_features, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_gates)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """ Produce the gate corrections of one episode.

        Args:
            embedding (torch.Tensor): Task embedding of shape [in_features].

        Returns:
            torch.Tensor: Corrections of shape [num_gates].
        """
        if self.input_norm == "l2":
            embedding = embedding / embedding.norm().clamp_min(1e-12)
        return self.out(F.relu(self.hidden(self.norm(embedding))))

    def config(self) -> Dict[str, Any]:
        """ Report the architecture so that it can be rebuilt on load.

        Returns:
            Dict[str, Any]: GateNet architecture.
        """
        return {
            "in_features": self.in_features,
            "hidden_size": self.hidden_size,
            "num_gates": self.num_gates,
            "input_norm": self.input_norm,
        }


def summarize_gates(gate_logits: np.ndarray) -> Dict[str, float]:
    """ Summarize the gates realized over a set of episodes.

    The reported variability is the spread of one gate across tasks, averaged
    and maximized over tensors. The spread between different tensors is a
    different quantity and is deliberately not reported as task variability: a
    model whose gates differ between layers but never move between tasks is not
    task-conditioned at all.

    Args:
        gate_logits (np.ndarray): Realized gate logits with shape
            [num_tasks, num_tensors].

    Returns:
        Dict[str, float]: Summary statistics.
    """
    logits = np.asarray(gate_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.size == 0:
        raise ValueError(
            "Gate summaries need a non-empty [num_tasks, num_tensors] matrix, "
            f"received shape {logits.shape}.")
    gates = 1.0 / (1.0 + np.exp(-logits))
    across_task_std = gates.std(axis=0)
    across_task_range = gates.max(axis=0) - gates.min(axis=0)
    return {
        "tasks": int(gates.shape[0]),
        "tensors": int(gates.shape[1]),
        "gate_mean": float(gates.mean()),
        "gate_min": float(gates.min()),
        "gate_max": float(gates.max()),
        "across_task_std_mean": float(across_task_std.mean()),
        "across_task_std_max": float(across_task_std.max()),
        "across_task_range_max": float(across_task_range.max()),
        "across_tensor_std": float(gates.mean(axis=0).std()),
    }


def format_gate_summary(summary: Dict[str, float],
                        iteration: Optional[int] = None) -> str:
    """ Render a gate summary as one line for the run log.

    Args:
        summary (Dict[str, float]): Output of ``summarize_gates``.
        iteration (Optional[int], optional): Meta-training iteration. Defaults
            to None.

    Returns:
        str: Single line summary.
    """
    prefix = "[tcsgmaml] gates"
    if iteration is not None:
        prefix = f"{prefix} @ iteration {iteration}"
    return (f"{prefix} | tasks {summary['tasks']} "
            f"| tensors {summary['tensors']} "
            f"| mean {summary['gate_mean']:.4f} "
            f"| min {summary['gate_min']:.4f} "
            f"| max {summary['gate_max']:.4f} "
            f"| across-task std mean {summary['across_task_std_mean']:.5f} "
            f"| across-task std max {summary['across_task_std_max']:.5f} "
            f"| across-task range max {summary['across_task_range_max']:.5f}")
