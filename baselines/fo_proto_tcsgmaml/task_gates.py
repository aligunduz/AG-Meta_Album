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


class TaskGates(nn.Module):
    """Persistent outer parameters; no task-specific tensors are stored here."""
    def __init__(self, encoder, config):
        super().__init__()
        weights = list(encoder.parameters())
        self.encoder_names = tuple(name for name, _ in encoder.named_parameters())
        self.shared_logits = nn.ParameterList([
            nn.Parameter(w.new_tensor(float(config["gate_init_logit"])))
            for w in weights
        ])
        gate_config = config["gate_net"]
        self.delta_scale = float(gate_config["delta_scale"])
        # Preserve the baseline's random stream when introducing the GateNet.
        # Linear modules are initialized on CPU before being moved to the device.
        with torch.random.fork_rng(devices=[]):
            self.gate_net = GateNet(
                encoder.in_features, gate_config["hidden_size"], len(weights),
                gate_config["input_norm"]).to(weights[0])

    def forward(self, task_embedding):
        deltas = self.gate_net(task_embedding) * self.delta_scale
        return torch.sigmoid(torch.stack(list(self.shared_logits)) + deltas)

    def architecture(self):
        return dict(gate_net=self.gate_net.config(), delta_scale=self.delta_scale,
                    encoder_names=list(self.encoder_names))
