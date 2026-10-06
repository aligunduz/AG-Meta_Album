"""One learned, unbounded inner step size per encoder parameter tensor.

    alpha[s, j] = encoder_lr * exp(log_scale[s, j])
    theta_next[j] = theta[j] - alpha[s, j] * g[j]

log_scale is zero at initialization, so the first adaptation is exactly
FO-Proto-MAML. There is no upper bound: the multiplier exp(log_scale) may grow
far beyond 1. With per_step=false (default) one row is shared by all inner
steps; with per_step=true every inner step owns a row.
"""
import torch
from torch import nn


class LayerwiseInnerLR(nn.Module):
    def __init__(self, encoder, config, encoder_lr, inner_steps):
        super().__init__()
        self.per_step = config["per_step"]
        if type(self.per_step) is not bool:
            raise ValueError("inner_lr.per_step must be boolean")
        if type(inner_steps) is not int or inner_steps < 1:
            raise ValueError("Learned inner step sizes require inner_steps >= 1")
        if type(encoder_lr) not in (int, float) or not 0 < encoder_lr < float("inf"):
            raise ValueError("encoder_lr must be a positive finite number")
        parameters = list(encoder.named_parameters())
        if not parameters or any(name.startswith("model.out.") for name, _ in parameters):
            raise ValueError("Pass an encoder with its registered classifier removed")
        self.names = [name for name, _ in parameters]
        self.shapes = [list(p.shape) for _, p in parameters]
        self.inner_steps = inner_steps
        self.base = float(encoder_lr)
        rows = inner_steps if self.per_step else 1
        # Zeros: no RNG consumed, multiplier exactly 1.
        self.log_scale = nn.Parameter(parameters[0][1].new_zeros((rows, len(self.names))))

    def rates(self):
        """Step sizes with shape (inner_steps, number of encoder tensors)."""
        return (self.base * self.log_scale.exp()).expand(self.inner_steps, -1)

    def layout(self):
        """Checkpoint manifest: ordered encoder tensors and the parametrization."""
        return dict(names=self.names, shapes=self.shapes, inner_steps=self.inner_steps,
                    per_step=self.per_step, base_lr=self.base,
                    parametrization="base_lr * exp(log_scale)")

    def validate_encoder(self, encoder, inner_steps):
        if (self.names != [name for name, _ in encoder.named_parameters()]
                or self.shapes != [list(p.shape) for p in encoder.parameters()]
                or self.inner_steps != inner_steps):
            raise ValueError("Inner step size encoder ordering/shape/step mismatch")

    @torch.no_grad()
    def metrics(self):
        """Multipliers exp(log_scale); geometric means are unweighted by tensor size."""
        log_scale = self.log_scale.detach().double()
        # Convolution/linear weights have ndim >= 2; normalization affine vectors ndim 1.
        matrix = torch.tensor([len(shape) >= 2 for shape in self.shapes],
                              device=log_scale.device)
        values = {"inner_lr/scale_min": log_scale.min().exp().item(),
                  "inner_lr/scale_geomean": log_scale.mean().exp().item(),
                  "inner_lr/scale_max": log_scale.max().exp().item()}
        for label, mask in (("weight", matrix), ("vector", ~matrix)):
            if mask.any():
                values[f"inner_lr/scale_geomean_{label}"] = log_scale[:, mask].mean().exp().item()
        if self.per_step:
            values.update({f"inner_lr/scale_geomean_step_{i + 1}": row.mean().exp().item()
                           for i, row in enumerate(log_scale)})
        return values

    @torch.no_grad()
    def scales(self):
        """Per-tensor multipliers, {name: [one value per stored row]}."""
        scale = self.log_scale.detach().double().exp().T.cpu().tolist()
        return dict(zip(self.names, scale, strict=True))
