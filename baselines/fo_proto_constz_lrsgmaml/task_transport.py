"""Shared LRSG bases conditioned only on a learned, episode-independent z."""
import math

import torch
from torch import nn

from gate_net import GateNet
from low_rank_transport import LowRankTransport


def validate_constant_condition(config):
    constant = config["constant_condition"]
    if constant["enabled"] is not True or constant["init"] != "zero":
        raise ValueError('constant_condition requires enabled=true and init="zero"')
    required = dict(enabled=True, hidden_size=128, input_norm="none",
                    scalar_delta_scale=0, low_rank_delta_scale=1)
    if any(config["task_conditioning"][key] != value for key, value in required.items()):
        raise ValueError("ConstZ requires the fixed LR-only GateNet configuration")
    if (config["lrsg"]["enabled"] is not True or config["lrsg"]["rank"] != 4
            or config["lrsg"]["beta"] != 1):
        raise ValueError("ConstZ requires enabled LRSG with rank=4 and beta=1")


class ConstantConditionedTransport(LowRankTransport):
    def __init__(self, encoder, config):
        validate_constant_condition(config)
        if encoder.in_features != 512:
            raise ValueError("ConstZ requires a 512-dimensional encoder embedding")
        super().__init__(encoder, config["lrsg"])
        # Registered outer parameter: saved and optimized with transport, but
        # never passed among the encoder/head fast weights. No RNG is consumed.
        self.z = nn.Parameter(next(encoder.parameters()).new_zeros(512))
        cfg = config["task_conditioning"]
        self.conditioning_enabled = cfg["enabled"]
        if type(self.conditioning_enabled) is not bool:
            raise ValueError("task_conditioning.enabled must be boolean")
        if self.conditioning_enabled and not self.enabled:
            raise ValueError("Task conditioning requires lrsg.enabled")
        self.scalar_scale = float(cfg["scalar_delta_scale"])
        self.low_rank_scale = float(cfg["low_rank_delta_scale"])
        if not all(math.isfinite(s) for s in (self.scalar_scale, self.low_rank_scale)):
            raise ValueError("Task residual scales must be finite")
        if type(cfg["hidden_size"]) is not int or cfg["hidden_size"] < 1:
            raise ValueError("hidden_size must be a positive integer")
        # Absolute output slices, in encoder registration order; never sort
        # numeric ParameterDict keys lexicographically.
        self.rank_layout = []
        offset = len(self.names)
        for name in self.names:
            key = self.indices[name]
            if key in self.u:
                rank = self.u[key].shape[1]
                self.rank_layout.append(dict(name=name, key=key, rank=rank,
                                             start=offset, stop=offset + rank))
                offset += rank
        self.output_size = offset
        self.gate_net = None
        if self.conditioning_enabled:
            # Same GateNet construction and RNG isolation as FO-Proto-TCSGMAML.
            with torch.random.fork_rng(devices=[]):
                self.gate_net = GateNet(encoder.in_features, cfg["hidden_size"],
                                        offset, cfg["input_norm"]).to(next(encoder.parameters()))

    def architecture(self):
        return dict(**super().architecture(),
                    constant_condition=dict(enabled=True, init="zero", shape=[512]),
                    task_conditioning=dict(enabled=self.conditioning_enabled,
                        gate_net=None if self.gate_net is None else self.gate_net.config(),
                        scalar_delta_scale=self.scalar_scale,
                        low_rank_delta_scale=self.low_rank_scale,
                        scalar_names=list(self.names), rank_layout=self.rank_layout,
                        output_size=self.output_size))

    def condition(self):
        # No episode argument and no detach: query loss must reach z through
        # GateNet and the transported first-order fast updates.
        residuals = self.gate_net(self.z)
        delta_a = residuals[:len(self.names)] * self.scalar_scale
        delta_c = {row["key"]: residuals[row["start"]:row["stop"]] * self.low_rank_scale
                   for row in self.rank_layout}
        if self.training:
            self._record_conditioning(delta_a, residuals[len(self.names):] * self.low_rank_scale)
        # Ephemeral tensors returned to adapt; no task graph is kept on module.
        return delta_a, delta_c

    def reset_metrics(self):
        super().reset_metrics()
        self._tc_sums = {}
        self._tc_counts = {}
        self._task_sum = {}
        self._task_square_sum = {}
        self._task_count = 0
        self._gate_stats = None

    @torch.no_grad()
    def _record_conditioning(self, delta_a, delta_c):
        self._task_count += 1
        for name, values in (("scalar_delta", delta_a), ("low_rank_delta", delta_c)):
            if not values.numel():
                continue
            values = values.detach().double()
            moments = torch.stack((values.sum(), values.square().sum(), values.abs().sum()))
            self._tc_sums[name] = self._tc_sums.get(name, 0) + moments
            self._tc_counts[name] = self._tc_counts.get(name, 0) + values.numel()
            # Per-coordinate task variance avoids confusing differences across
            # layers/ranks with actual changes from one episode to another.
            self._task_sum[name] = self._task_sum.get(name, 0) + values
            self._task_square_sum[name] = self._task_square_sum.get(name, 0) + values.square()
        gates = (torch.stack(list(self.logits.values())) + delta_a).sigmoid()
        current = torch.stack((gates.sum(), gates.min(), gates.max(), gates.new_tensor(gates.numel())))
        if self._gate_stats is None:
            self._gate_stats = current
        else:
            self._gate_stats[0] += current[0]
            self._gate_stats[1] = torch.minimum(self._gate_stats[1], current[1])
            self._gate_stats[2] = torch.maximum(self._gate_stats[2], current[2])
            self._gate_stats[3] += current[3]

    @torch.no_grad()
    def metrics(self):
        values = super().metrics()
        if self._gate_stats is not None:
            total, minimum, maximum, count = self._gate_stats.tolist()
            values.update({"lrsg/scalar_gate_mean": total / count,
                           "lrsg/scalar_gate_min": minimum, "lrsg/scalar_gate_max": maximum})
        for name in ("scalar_delta", "low_rank_delta"):
            mean = std = absolute = task_std = 0.0
            if name in self._tc_sums:
                mean, second, absolute = (self._tc_sums[name] / self._tc_counts[name]).tolist()
                std = math.sqrt(max(0.0, second - mean * mean))
                variance = (self._task_square_sum[name] / self._task_count
                            - (self._task_sum[name] / self._task_count).square())
                task_std = variance.clamp_min(0).sqrt().mean().item()
            values.update({f"tc/{name}_mean": mean, f"tc/{name}_std": std,
                           f"tc/{name}_abs_mean": absolute,
                           f"tc/{name}_task_std_mean": task_std})
        values["tc/low_rank_coeff_mean"] = 1 + values["tc/low_rank_delta_mean"]
        values["tc/low_rank_coeff_std"] = values["tc/low_rank_delta_std"]
        return values
