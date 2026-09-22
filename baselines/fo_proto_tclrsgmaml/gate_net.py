"""TCSGMAML's GateNet and shared scalar logits, without a task encoder."""
import torch
from torch import nn
from torch.nn import functional as F


class GateNet(nn.Module):
    """Same Linear-ReLU-Linear architecture and zero output init as TCSGMAML."""
    def __init__(self, in_features, hidden_size, num_gates, input_norm="none"):
        super().__init__()
        if input_norm not in ("none", "l2", "layernorm"):
            raise ValueError("Unsupported gate_net.input_norm")
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

    def forward(self, embedding):
        # Optional normalization is LOCAL to the GateNet input, exactly as in
        # TCSGMAML. The controlled config uses 'none'; prototypes never use it.
        if self.input_norm == "l2":
            embedding = embedding / embedding.norm().clamp_min(1e-12)
        return self.out(F.relu(self.hidden(self.norm(embedding))))

    def config(self):
        return dict(in_features=self.in_features, hidden_size=self.hidden_size,
                    num_gates=self.num_gates, input_norm=self.input_norm)
