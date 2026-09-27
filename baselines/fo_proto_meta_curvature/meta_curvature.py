"""Encoder-only MC2, with explicit output/input/filter mode matrices.

G'[o,i,f] = sum_{a,b,c} Mo[o,a] Mi[i,b] Mf[f,c] G[a,b,c].
For the official TF implementation: Mo = mc_out.T, Mi = mc_in,
Mf = mc_f. Linear weights use G' = Mo @ G @ Mi.T.
"""
import torch
from torch import nn


class ParameterCurvature(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.shape = tuple(parameter.shape)
        options = dict(device=parameter.device, dtype=parameter.dtype)
        if parameter.ndim == 1:
            self.scale = nn.Parameter(torch.ones(self.shape, **options))
        elif parameter.ndim in (2, 4):
            self.Mo = nn.Parameter(torch.eye(self.shape[0], **options))
            self.Mi = nn.Parameter(torch.eye(self.shape[1], **options))
            if parameter.ndim == 4:
                self.Mf = nn.Parameter(torch.eye(self.shape[2] * self.shape[3], **options))
        else:
            raise ValueError(f"Unsupported encoder parameter shape: {self.shape}")

    def forward(self, gradient):
        if tuple(gradient.shape) != self.shape:
            raise ValueError(f"Expected gradient shape {self.shape}, got {tuple(gradient.shape)}")
        if gradient.ndim == 1:
            return self.scale * gradient
        if gradient.ndim == 2:
            return self.Mo @ gradient @ self.Mi.T
        g = gradient.flatten(2)
        # Separate mode products; never materialize the full Kronecker matrix.
        g = torch.einsum("oa,aif->oif", self.Mo, g)
        g = torch.einsum("ib,obf->oif", self.Mi, g)
        g = torch.einsum("fc,oic->oif", self.Mf, g)
        return g.reshape(self.shape)


class EncoderCurvature(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        parameters = list(encoder.named_parameters())
        if not parameters or any(name.startswith("model.out.") for name, _ in parameters):
            raise ValueError("Pass an encoder with its registered classifier removed")
        self.names = tuple(name for name, _ in parameters)
        self.transforms = nn.ModuleList([ParameterCurvature(p) for _, p in parameters])

    def architecture(self):
        """Checkpoint manifest maps each ordered encoder name to its MC tensors."""
        return [dict(name=name, shape=list(transform.shape),
                     curvature={key: list(p.shape) for key, p in transform.named_parameters()})
                for name, transform in zip(self.names, self.transforms)]

    def forward(self, gradients):
        if len(gradients) != len(self.transforms):
            raise ValueError("Curvature accepts exactly the encoder gradients; exclude the head")
        return [transform(g) for transform, g in zip(self.transforms, gradients)]
