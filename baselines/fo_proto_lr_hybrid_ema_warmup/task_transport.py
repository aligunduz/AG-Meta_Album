"""Shared LRSG bases conditioned on learned z, EMA or past training tasks."""
import math

import torch
from torch import nn

from gate_net import GateNet
from low_rank_transport import LowRankTransport


def validate_constant_condition(config):
    constant = config["constant_condition"]
    if constant["enabled"] is not True or constant["init"] not in ("zero", "ema", "shuffle"):
        raise ValueError('constant_condition requires enabled=true and init="zero", "ema" or "shuffle"')
    if constant["init"] in ("ema", "shuffle"):
        decay = constant.get("ema_decay")
        if (type(decay) not in (int, float) or not math.isfinite(decay)
                or not 0 <= decay < 1):
            raise ValueError("constant_condition.ema_decay must be finite and in [0, 1)")
    warmup_keys = ("ema_warmup_alpha", "ema_warmup_tasks")
    if any(key in constant for key in warmup_keys):
        if not all(key in constant for key in warmup_keys):
            raise ValueError("ema_warmup_alpha and ema_warmup_tasks must appear together")
        if constant["init"] != "ema":
            raise ValueError('EMA warm-up is only valid for init="ema"')
        alpha = constant["ema_warmup_alpha"]
        if (type(alpha) not in (int, float) or not math.isfinite(alpha)
                or not 0 < alpha <= 1):
            raise ValueError("ema_warmup_alpha must be finite and in (0, 1]")
        tasks = constant["ema_warmup_tasks"]
        if type(tasks) is not int or tasks < 1:
            raise ValueError("ema_warmup_tasks must be a positive integer")
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
        if config["constant_condition"]["init"] != "ema" or "ema_warmup_tasks" not in config["constant_condition"]:
            raise ValueError("Hybrid requires EMA with warmup")
        if encoder.in_features != 512:
            raise ValueError("ConstZ requires a 512-dimensional encoder embedding")
        super().__init__(encoder, config["lrsg"])
        self.alpha = config["hybrid"]["alpha"]
        if type(self.alpha) not in (int, float) or not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("hybrid.alpha must be finite in [0, 1]")
        self.capture = config.get("embedding_logging", {}).get("enabled", True)
        self.observation = None
        self.init_mode = config["constant_condition"]["init"]
        self.ema_warmup_tasks = config["constant_condition"].get("ema_warmup_tasks")
        reference = next(encoder.parameters())
        if self.ema_warmup_tasks is not None:
            self.ema_warmup_alpha = float(config["constant_condition"]["ema_warmup_alpha"])
            # Only the opt-in schedule adds state; legacy checkpoint keys stay
            # unchanged. Count completed training tasks, never optimizer steps.
            self.register_buffer("ema_completed_tasks",
                                 torch.tensor(0, dtype=torch.long, device=reference.device))
        if self.init_mode == "zero":
            # Preserve the original parameter/state layout and optimization.
            self.z = nn.Parameter(reference.new_zeros(512))
        else:
            self.ema_decay = float(config["constant_condition"]["ema_decay"])
            self.register_buffer("m", reference.new_zeros(512))
            self.register_buffer("m_initialized", torch.tensor(False, device=reference.device))
        if self.init_mode == "shuffle":
            # Training-only FIFO ring: follows module device/dtype, but is not
            # saved with the inference checkpoint. Entries never retain graphs.
            self.register_buffer("_shuffle_embeddings", reference.new_zeros(256, 512),
                                 persistent=False)
            self._shuffle_count = 0
            self._shuffle_next = 0
            # Isolate sampling from model/data RNG, matching model seed 98.
            self._shuffle_generator = torch.Generator(device="cpu").manual_seed(98)
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
        if self.init_mode in ("ema", "shuffle"):
            constant["ema_decay"] = self.ema_decay
        if self.ema_warmup_tasks is not None:
            constant.update(ema_warmup_alpha=self.ema_warmup_alpha,
                            ema_warmup_tasks=self.ema_warmup_tasks)
        if self.init_mode == "shuffle":
            constant["shuffle_capacity"] = 256
        return dict(**super().architecture(),
                    constant_condition=constant, hybrid=dict(alpha=self.alpha),
                    task_conditioning=dict(enabled=self.conditioning_enabled,
                        gate_net=None if self.gate_net is None else self.gate_net.config(),
                        scalar_delta_scale=self.scalar_scale,
                        low_rank_delta_scale=self.low_rank_scale,
                        scalar_names=list(self.names), rank_layout=self.rank_layout,
                        output_size=self.output_size))

    def condition(self, task_embedding):
        task = task_embedding.detach()
        initialized = bool(self.m_initialized)
        previous = self.m.detach().clone()
        hybrid = self.alpha * task + (1 - self.alpha) * previous if initialized else task
        residuals = self.gate_net(hybrid) if initialized else self._zero_residuals()
        delta_a = residuals[:len(self.names)] * self.scalar_scale
        delta_c = {row["key"]: residuals[row["start"]:row["stop"]] * self.low_rank_scale
                   for row in self.rank_layout}
        if self.training:
            self._record_conditioning(delta_a, residuals[len(self.names):] * self.low_rank_scale)
        if self.capture:
            warmup = self.ema_completed_tasks.item() < self.ema_warmup_tasks
            self.observation = dict(
                vectors={"z_task": task, "z_ema_before": previous, "z_hybrid": hybrid,
                         "gate_input": hybrid, "delta_c_tau": residuals[len(self.names):]},
                alpha=self.alpha, ema_initialized=initialized, fallback=not initialized,
                gate_bypassed=not initialized, warmup=warmup,
                ema_update_enabled=self.training,
                effective_update_decay=(0.0 if not initialized else (1-self.ema_warmup_alpha if warmup else self.ema_decay)) if self.training else None,
                ema_decay=1-self.ema_warmup_alpha if warmup else self.ema_decay,
                ema_completed_tasks=self.ema_completed_tasks.item())
            self.observation["vectors"] = {k: v.detach().cpu().float().numpy().copy()
                                            for k, v in self.observation["vectors"].items()}
            self.episode_corrections = []
        return delta_a, delta_c

    def transport_gradient(self, name, grad, conditioning=None):
        result = super().transport_gradient(name, grad, conditioning)
        if self.capture:
            with torch.no_grad():
                key = self.indices[name]
                if key in self.u:
                    scalar = (self.logits[key] + conditioning[0][int(key)]).sigmoid()
                    correction = (result - scalar * grad.detach()).norm().item()
                    norm = grad.norm().item()
                    self.episode_corrections.append((correction, norm,
                        correction / norm if norm > 1e-12 else None))
        return result

    def _zero_residuals(self):
        # First episode: bypass GateNet entirely, even with learned biases.
        # Connected zeros preserve the outer loop's no-unused-param contract.
        zero = sum(p.sum() * 0 for p in self.gate_net.parameters())
        return self.m.new_zeros(self.output_size) + zero

    @torch.no_grad()
    def complete_training_episode(self, task_embedding):
        """Publish an embedding only after adaptation/query/backward complete."""
        if self.init_mode == "zero" or not self.training:
            return
        self.update_ema(task_embedding)
        if self.ema_warmup_tasks is not None:
            self.ema_completed_tasks.add_(1)
        if self.init_mode == "shuffle":
            self._shuffle_embeddings[self._shuffle_next].copy_(task_embedding.detach())
            self._shuffle_next = (self._shuffle_next + 1) % 256
            self._shuffle_count = min(self._shuffle_count + 1, 256)

    @torch.no_grad()
    def update_ema(self, task_embedding):
        """Commit a completed training episode's detached initial support mean.

        Called through complete_training_episode after query/backward and the
        outer bookkeeping. Shuffle maintains this EMA for validation/test.
        Adaptation and evaluation never commit episode information here.
        """
        if self.init_mode not in ("ema", "shuffle") or not self.training:
            return
        if task_embedding.shape != self.m.shape:
            raise ValueError("EMA task embedding must have shape [512]")
        embedding = task_embedding.detach().to(self.m)
        if not bool(self.m_initialized):
            self.m.copy_(embedding)
            self.m_initialized.fill_(True)
        elif (self.ema_warmup_tasks is not None
              and self.ema_completed_tasks.item() < self.ema_warmup_tasks):
            # Counter excludes this task until complete_training_episode returns:
            # task 5000 sees 4999; task 5001 sees 5000. Task 1 always copies above.
            self.m.mul_(1 - self.ema_warmup_alpha).add_(embedding, alpha=self.ema_warmup_alpha)
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
