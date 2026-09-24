"""One bounded scalar per inner step and encoder parameter tensor."""
import math

import torch
from torch import nn


class LearnedInnerLR(nn.Module):
    def __init__(self, encoder, inner_steps):
        super().__init__()
        self.names = [name for name, _ in encoder.named_parameters()]
        self.shapes = [list(p.shape) for p in encoder.parameters()]
        self.inner_steps = inner_steps
        reference = next(encoder.parameters())
        # sigmoid(log(9/40)) = (.01 - .001) / .049. No RNG consumed.
        self.logits = nn.Parameter(reference.new_full(
            (inner_steps, len(self.names)), math.log(9 / 40)))

    def rates(self):
        return .001 + .049 * self.logits.sigmoid()

    def layout(self):
        return dict(names=self.names, shapes=self.shapes, inner_steps=self.inner_steps,
                    minimum=.001, maximum=.05, initial=.01)

    def validate_encoder(self, encoder, inner_steps):
        if (self.names != [name for name, _ in encoder.named_parameters()]
                or self.shapes != [list(p.shape) for p in encoder.parameters()]
                or self.inner_steps != inner_steps):
            raise ValueError("Inner LR encoder ordering/shape/step mismatch")

    @torch.no_grad()
    def metrics(self):
        rates = self.rates()
        values = {"inner_lr/min": rates.min().item(),
                  "inner_lr/mean": rates.mean().item(),
                  "inner_lr/max": rates.max().item()}
        values.update({f"inner_lr/step_{i + 1}_mean": value.item()
                       for i, value in enumerate(rates.mean(dim=1))})
        return values
