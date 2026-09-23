"""Shared LRSG bases conditioned on learned z or past training episodes' EMA."""
import math

import torch
from torch import nn

from gate_net import GateNet
from low_rank_transport import LowRankTransport


def validate_constant_condition(config):
    constant = config["constant_condition"]
    if constant["enabled"] is not True or constant["init"] not in ("zero", "ema"):
        raise ValueError('constant_condition requires enabled=true and init="zero" or "ema"')
    if constant["init"] == "ema":
        decay = constant.get("ema_decay")
        if (type(decay) not in (int, float) or not math.isfinite(decay)
                or not 0 <= decay < 1):
            raise ValueError("constant_condition.ema_decay must be finite and in [0, 1)")
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
        self.init_mode = config["constant_condition"]["init"]
        reference = next(encoder.parameters())
        if self.init_mode == "zero":
            # Preserve the original parameter/state layout and optimization.
            self.z = nn.Parameter(reference.new_zeros(512))
        else:
            self.ema_decay = float(config["constant_condition"]["ema_decay"])
            self.register_buffer("m", reference.new_zeros(512))
            self.register_buffer("m_initialized", torch.tensor(False, device=reference.device))
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
        constant = dict(enabled=True, init=self.init_mode, shape=[512])
        if self.init_mode == "ema":
            constant["ema_decay"] = self.ema_decay
        return dict(**super().architecture(),
                    constant_condition=constant,
                    task_conditioning=dict(enabled=self.conditioning_enabled,
                        gate_net=None if self.gate_net is None else self.gate_net.config(),
                        scalar_delta_scale=self.scalar_scale,
                        low_rank_delta_scale=self.low_rank_scale,
                        scalar_names=list(self.names), rank_layout=self.rank_layout,
                        output_size=self.output_size))

    def condition(self):
        # No current-episode argument. Clone the buffer to give this episode
        # its own immutable input even if an EMA update occurs before backward.
        if self.init_mode == "zero":
            residuals = self.gate_net(self.z)
        elif bool(self.m_initialized):
            residuals = self.gate_net(self.m.detach().clone())
        else:
            # First episode: bypass GateNet entirely, even with learned biases.
            # Connected zeros preserve the outer loop's no-unused-param contract.
            zero = sum(p.sum() * 0 for p in self.gate_net.parameters())
            residuals = self.m.new_zeros(self.output_size) + zero
        delta_a = residuals[:len(self.names)] * self.scalar_scale
        delta_c = {row["key"]: residuals[row["start"]:row["stop"]] * self.low_rank_scale
                   for row in self.rank_layout}
        if self.training:
            self._record_conditioning(delta_a, residuals[len(self.names):] * self.low_rank_scale)
        # Ephemeral tensors returned to adapt; no task graph is kept on module.
        return delta_a, delta_c

    @torch.no_grad()
    def update_ema(self, task_embedding):
        """Commit a completed training episode's detached initial support mean.

        Called by meta_fit only after query/backward and the outer bookkeeping.
        Adaptation and evaluation never commit episode information here.
        """
        if self.init_mode != "ema" or not self.training:
            return
        if task_embedding.shape != self.m.shape:
            raise ValueError("EMA task embedding must have shape [512]")
        embedding = task_embedding.detach().to(self.m)
        if not bool(self.m_initialized):
            self.m.copy_(embedding)
            self.m_initialized.fill_(True)
        else:
            self.m.mul_(self.ema_decay).add_(embedding, alpha=1 - self.ema_decay)

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
