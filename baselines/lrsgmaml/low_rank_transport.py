"""Static, meta-learned Conv2d gradient factors and detached diagnostics."""

import math

import torch
from torch import nn


class LowRankTransport(nn.Module):
    """Apply an output-channel low-rank correction to Conv2d gradients.

    Factors follow real parameter names, not assumed fast-weight offsets.
    Creating factors uses a private CPU generator and never advances the
    global CPU or CUDA RNG. Only task fast weights are adapted in the inner
    loop; these shared factors belong to the outer optimizer.
    """

    FORMAT_VERSION = 1
    TRANSPORT_TYPE = "conv2d_output_channel_low_rank"

    def __init__(self, model, rank=4, seed=98):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("low_rank.rank must be a positive integer.")
        self.rank = rank
        named_parameters = dict(model.named_parameters())
        self.weight_names = tuple(named_parameters)
        conv_names = set()
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                name = f"{module_name}.weight" if module_name else "weight"
                if name not in named_parameters:
                    raise ValueError(f"Conv2d weight '{name}' is missing from named_parameters().")
                conv_names.add(name)
        self.conv_names = tuple(name for name in self.weight_names if name in conv_names)
        self.name_to_key = {name: f"conv_{index}" for index, name in enumerate(self.conv_names)}
        self.weight_shapes = {name: tuple(named_parameters[name].shape) for name in self.conv_names}
        self.effective_ranks = {
            name: min(rank, self.weight_shapes[name][0]) for name in self.conv_names
        }
        self.U = nn.ParameterDict()
        self.V = nn.ParameterDict()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for name in self.conv_names:
            weight = named_parameters[name]
            out_channels = weight.shape[0]
            shape = (out_channels, self.effective_ranks[name])
            key = self.name_to_key[name]
            self.U[key] = nn.Parameter(torch.zeros(shape, device=weight.device, dtype=weight.dtype))
            initial_v = torch.randn(shape, generator=generator, device="cpu", dtype=weight.dtype)
            initial_v = initial_v * (1.0 / math.sqrt(out_channels))
            self.V[key] = nn.Parameter(initial_v.to(device=weight.device, dtype=weight.dtype))

    def validate_fast_weights(self, weight_names, weights):
        """Check order and Conv2d shapes while permitting any-way classifiers."""
        if tuple(weight_names) != self.weight_names:
            raise ValueError("Fast-weight names/order do not match the low-rank transport model.")
        if len(weights) != len(self.weight_names):
            raise ValueError("Fast-weight count does not match the low-rank transport model.")
        for name, weight in zip(self.weight_names, weights):
            if name in self.weight_shapes and tuple(weight.shape) != self.weight_shapes[name]:
                raise ValueError(
                    f"Conv2d fast weight '{name}' has shape {tuple(weight.shape)}; "
                    f"expected {self.weight_shapes[name]}."
                )

    def correction(self, name, clipped_grad):
        """Return U @ (V.T @ G), without forming a Cout-by-Cout matrix."""
        if name not in self.name_to_key or clipped_grad is None:
            return None
        if tuple(clipped_grad.shape) != self.weight_shapes[name]:
            raise ValueError(f"Gradient shape for Conv2d '{name}' does not match its weight.")
        key = self.name_to_key[name]
        gradient = clipped_grad.reshape(self.weight_shapes[name][0], -1)
        return (self.U[key] @ (self.V[key].transpose(0, 1) @ gradient)).reshape_as(clipped_grad)

    def dump_state(self):
        """Return an independent detached snapshot, including its schema."""
        return {
            "format_version": self.FORMAT_VERSION,
            "transport_type": self.TRANSPORT_TYPE,
            "rank": self.rank,
            "weight_names": list(self.weight_names),
            "conv_names": list(self.conv_names),
            "layers": {
                name: {
                    "weight_shape": list(self.weight_shapes[name]),
                    "rank": self.effective_ranks[name],
                    "U": self.U[self.name_to_key[name]].detach().cpu().clone(),
                    "V": self.V[self.name_to_key[name]].detach().cpu().clone(),
                }
                for name in self.conv_names
            },
        }

    def load_state(self, state, trainable=False):
        """Validate the entire checkpoint, then copy without random draws.

        Validation happens before mutation so an incompatible layer cannot
        leave a partially loaded transport. Copying retains Parameter objects
        and prevents checkpoint tensor storage from becoming model storage.
        """
        if not isinstance(state, dict):
            raise ValueError("Low-rank transport checkpoint must be a dictionary.")
        expected_fields = {
            "format_version", "transport_type", "rank", "weight_names", "conv_names", "layers"
        }
        if set(state) != expected_fields:
            raise ValueError("Low-rank transport checkpoint has missing or unexpected metadata fields.")
        if type(state["format_version"]) is not int or state["format_version"] != self.FORMAT_VERSION:
            raise ValueError("Unsupported low-rank transport checkpoint format version.")
        if state["transport_type"] != self.TRANSPORT_TYPE:
            raise ValueError("Low-rank transport checkpoint has an incompatible transformation type.")
        if type(state["rank"]) is not int or state["rank"] != self.rank:
            raise ValueError(f"Low-rank transport checkpoint rank must be {self.rank}.")
        for field, expected in (("weight_names", self.weight_names), ("conv_names", self.conv_names)):
            names = state[field]
            if not isinstance(names, (list, tuple)) or tuple(names) != expected:
                raise ValueError(f"Low-rank transport checkpoint {field} do not match model names/order.")
        layers = state["layers"]
        if not isinstance(layers, dict) or set(layers) != set(self.conv_names):
            raise ValueError("Low-rank transport checkpoint has missing or unexpected Conv2d parameter names.")
        for name in self.conv_names:
            layer = layers[name]
            if not isinstance(layer, dict) or set(layer) != {"weight_shape", "rank", "U", "V"}:
                raise ValueError(f"Low-rank transport checkpoint layer '{name}' has invalid fields.")
            shape = layer["weight_shape"]
            if not isinstance(shape, (list, tuple)) or tuple(shape) != self.weight_shapes[name]:
                raise ValueError(f"Low-rank transport checkpoint weight shape mismatch for '{name}'.")
            if type(layer["rank"]) is not int or layer["rank"] != self.effective_ranks[name]:
                raise ValueError(f"Low-rank transport checkpoint effective rank mismatch for '{name}'.")
            expected_shape = (self.weight_shapes[name][0], self.effective_ranks[name])
            for factor_name in ("U", "V"):
                value = layer[factor_name]
                if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                        or tuple(value.shape) != expected_shape):
                    raise ValueError(
                        f"Low-rank transport checkpoint {factor_name} for '{name}' "
                        f"must be a floating tensor with shape {expected_shape}."
                    )
        with torch.no_grad():
            for name in self.conv_names:
                key = self.name_to_key[name]
                self.U[key].copy_(layers[name]["U"].detach())
                self.V[key].copy_(layers[name]["V"].detach())
                self.U[key].requires_grad_(trainable)
                self.V[key].requires_grad_(trainable)


class TransportDiagnostics:
    """Aggregate detached statistics from already-selected validation tasks.

    The caller bounds the tasks (normally the first three existing tasks).
    This collector stores scalar sums/counts only, never autograd graphs or
    batches, and does not sample data or consume RNG.
    """

    EPS = 1e-12

    def __init__(self, transport, gate_logits=None, max_tasks=3):
        self.transport = transport
        self.max_tasks = max_tasks
        self.gate_logits = None if gate_logits is None else [gate.detach() for gate in gate_logits]
        self.layers = {
            name: {
                "observations": 0,
                "finite_observations": 0,
                "nonfinite_observations": 0,
                "zero_input_norm": 0,
                "zero_scalar_norm": 0,
                "zero_transformed_norm": 0,
                "correction_ratio_sum": 0.0,
                "correction_ratio_max": None,
                "cosine_sum": 0.0,
                "cosine_count": 0,
            }
            for name in transport.conv_names
        }

    def record(self, name, clipped_grad, scalar_part, correction, transformed):
        if (name not in self.layers or correction is None or clipped_grad is None
                or scalar_part is None or transformed is None):
            return
        values = [value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
                  for value in (clipped_grad, scalar_part, correction, transformed)]
        stats = self.layers[name]
        stats["observations"] += 1
        if not all(bool(torch.isfinite(value).all()) for value in values):
            stats["nonfinite_observations"] += 1
            return
        input_grad, scalar_grad, residual, output_grad = values
        input_norm, scalar_norm, residual_norm, output_norm = [
            torch.linalg.vector_norm(value).item() for value in values
        ]
        if not all(math.isfinite(value) for value in (input_norm, scalar_norm, residual_norm, output_norm)):
            stats["nonfinite_observations"] += 1
            return
        stats["zero_input_norm"] += int(input_norm == 0.0)
        stats["zero_scalar_norm"] += int(scalar_norm == 0.0)
        stats["zero_transformed_norm"] += int(output_norm == 0.0)
        ratio = residual_norm / (scalar_norm + self.EPS)
        if not math.isfinite(ratio):
            stats["nonfinite_observations"] += 1
            return
        stats["finite_observations"] += 1
        stats["correction_ratio_sum"] += ratio
        previous_max = stats["correction_ratio_max"]
        stats["correction_ratio_max"] = ratio if previous_max is None else max(previous_max, ratio)
        # A zero vector has no direction. Exclude it from the cosine mean and
        # retain the explicit zero counts instead of reporting a spurious 1.
        if input_norm > 0.0 and output_norm > 0.0:
            cosine = torch.dot(input_grad / input_norm, output_grad / output_norm).item()
            if math.isfinite(cosine):
                stats["cosine_sum"] += max(-1.0, min(1.0, cosine))
                stats["cosine_count"] += 1

    @staticmethod
    def _finite_number(value):
        return float(value) if math.isfinite(value) else None

    def summary(self, gate_logits=None):
        """Return strict-JSON-compatible values; undefined measurements are null."""
        gates = self.gate_logits if gate_logits is None else gate_logits
        if gates is None or len(gates) != len(self.transport.weight_names):
            raise ValueError("Diagnostics require one scalar gate per fast-weight tensor.")
        gate_values = {}
        finite_gates = []
        for name, gate in zip(self.transport.weight_names, gates):
            if not isinstance(gate, torch.Tensor) or gate.ndim != 0:
                raise ValueError("Diagnostics gate logits must be scalar tensors.")
            value = torch.sigmoid(gate.detach()).item()
            gate_values[name] = self._finite_number(value)
            if math.isfinite(value):
                finite_gates.append(value)
        layers = {}
        for name, stats in self.layers.items():
            key = self.transport.name_to_key[name]
            u_norm = torch.linalg.vector_norm(self.transport.U[key].detach().double()).item()
            v_norm = torch.linalg.vector_norm(self.transport.V[key].detach().double()).item()
            finite_count = stats["finite_observations"]
            cosine_count = stats["cosine_count"]
            layers[name] = {
                "U_frobenius_norm": self._finite_number(u_norm),
                "V_frobenius_norm": self._finite_number(v_norm),
                "correction_ratio_mean": self._finite_number(stats["correction_ratio_sum"] / finite_count)
                if finite_count else None,
                "correction_ratio_max": stats["correction_ratio_max"],
                "gradient_cosine_mean": self._finite_number(stats["cosine_sum"] / cosine_count)
                if cosine_count else None,
                **{key: value for key, value in stats.items()
                   if key not in {"correction_ratio_sum", "correction_ratio_max", "cosine_sum"}},
            }
        return {
            "max_validation_tasks": self.max_tasks,
            "ratio_epsilon": self.EPS,
            "gate_statistics": {
                "count": len(gates),
                "finite_count": len(finite_gates),
                "nonfinite_count": len(gates) - len(finite_gates),
                "minimum": min(finite_gates) if finite_gates else None,
                "maximum": max(finite_gates) if finite_gates else None,
                "mean": sum(finite_gates) / len(finite_gates) if finite_gates else None,
                "per_parameter": gate_values,
            },
            "layers": layers,
        }
