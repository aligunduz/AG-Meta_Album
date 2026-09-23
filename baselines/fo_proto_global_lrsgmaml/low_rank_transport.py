"""LRSG with global, outer-only learnable residuals on the rank coefficients."""
import math

import torch
from torch import nn


class GlobalLowRankTransport(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.enabled = config["enabled"]
        self.rank = config["rank"]
        self.beta = float(config["beta"])
        if type(self.enabled) is not bool or type(self.rank) is not int or self.rank < 1:
            raise ValueError("lrsg requires boolean enabled and positive integer rank")
        logit = float(config.get("gate_init_logit", 4.0))
        if not math.isfinite(self.beta) or not math.isfinite(logit):
            raise ValueError("beta and gate_init_logit must be finite")
        named = list(encoder.named_parameters())
        self.names = [name for name, _ in named]
        self.shapes = [list(w.shape) for _, w in named]
        self.indices = {name: str(i) for i, name in enumerate(self.names)}
        self.logits = nn.ParameterDict()
        self.u = nn.ParameterDict()
        self.v = nn.ParameterDict()
        self.global_delta_c = nn.ParameterDict()
        if self.enabled:
            for name in self.names:
                self.logits[self.indices[name]] = nn.Parameter(named[0][1].new_tensor(logit))
            # Same initialization as baselines/lrsgmaml:
            # U = 0, V ~ N(0, 1/C_out)
            generator = torch.Generator(device="cpu").manual_seed(98)

            for name, weight in named:
                if name.split(".")[-1] == "weight" and weight.ndim >= 2:
                    key = self.indices[name]

                    shape = (
                        weight.shape[0],
                        min(self.rank, weight.shape[0])
                    )

                    initial_v = torch.randn(
                        shape,
                        generator=generator,
                        dtype=weight.dtype
                    )

                    self.u[key] = nn.Parameter(
                        torch.zeros(
                            shape,
                            dtype=weight.dtype,
                            device=weight.device
                        )
                    )

                    self.v[key] = nn.Parameter(
                        (initial_v * (1.0 / math.sqrt(weight.shape[0]))).to(weight)
                    )
                    # One shared coefficient vector per eligible encoder layer.
                    # Zero initialization preserves LRSG exactly and consumes no RNG.
                    self.global_delta_c[key] = nn.Parameter(weight.new_zeros(shape[1]))
        self.reset_metrics()

    def architecture(self):
        return dict(names=self.names, shapes=self.shapes, rank=self.rank,
                    enabled=self.enabled, beta=self.beta,
                    global_delta_c_layout=[
                        dict(name=name, key=self.indices[name],
                             rank=self.global_delta_c[self.indices[name]].numel())
                        for name in self.names if self.indices[name] in self.global_delta_c])

    def reset_metrics(self):
        self._sums = None
        self._count = 0

    def transport_gradient(self, name, grad):
        # Stop support Hessians; keep the outer graph in logits/U/V/global_delta_c.
        grad = grad.detach()
        if not self.enabled:
            return grad
        key = self.indices[name]
        transformed = self.logits[key].sigmoid() * grad
        if key in self.u:
            matrix = grad.reshape(grad.shape[0], -1)
            projected = self.v[key].T @ matrix
            projected = (1 + self.global_delta_c[key]).unsqueeze(1) * projected
            correction = self.beta * (self.u[key] @ projected)
            transformed = transformed + correction.reshape_as(grad)
            if self.training:
                with torch.no_grad():
                    original_norm = grad.norm()
                    correction_norm = correction.norm()
                    values = torch.stack((correction_norm, original_norm,
                                          correction_norm / (original_norm + 1e-12)))
                    self._sums = values if self._sums is None else self._sums + values
                    self._count += 1
        return transformed

    @torch.no_grad()
    def metrics(self):
        if not self.enabled:
            return {}
        gates = torch.stack(list(self.logits.values())).sigmoid()
        values = dict(scalar_gate_mean=gates.mean().item(),
                      scalar_gate_min=gates.min().item(), scalar_gate_max=gates.max().item())
        averages = ([0.0] * 3 if self._sums is None else
                    (self._sums / self._count).cpu().tolist())
        values.update(zip(("low_rank_correction_norm", "original_gradient_norm",
                           "correction_to_gradient_ratio"), averages))
        for name, params in (("u_norm", self.u), ("v_norm", self.v)):
            values[name] = (torch.stack([p.norm() for p in params.values()]).mean().item()
                            if len(params) else 0.0)
        if len(self.global_delta_c):
            coefficients = torch.cat([p.reshape(-1) for p in self.global_delta_c.values()])
            values.update(global_delta_c_mean=coefficients.mean().item(),
                          global_delta_c_mean_abs=coefficients.abs().mean().item(),
                          global_delta_c_norm=coefficients.norm().item(),
                          global_delta_c_max_abs=coefficients.abs().max().item())
        else:
            values.update(global_delta_c_mean=0.0, global_delta_c_mean_abs=0.0,
                          global_delta_c_norm=0.0, global_delta_c_max_abs=0.0)
        return {"lrsg/" + k: v for k, v in values.items()}
