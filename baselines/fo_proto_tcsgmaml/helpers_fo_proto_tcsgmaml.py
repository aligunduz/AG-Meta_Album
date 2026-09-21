"""Shared-Embedding FO-Proto-TCSGMAML functional adaptation.

Only update gradients are stopped. The support-derived head stays connected
to the original encoder throughout adaptation and the outer backward pass.
"""
import torch
import torch.nn.functional as F


def prototype_head(features, labels, num_classes=None):
    classes = torch.unique(labels, sorted=True)
    ways = classes.numel()
    if (labels.dtype != torch.long or labels.ndim != 1 or ways == 0
            or features.shape[0] != labels.numel()
            or not torch.equal(classes, torch.arange(ways, device=labels.device))
            or (num_classes is not None and num_classes != ways)):
        raise ValueError("Support labels must cover contiguous indices 0..way-1")
    prototypes = torch.stack([features[labels == k].mean(0) for k in range(ways)])
    return 2 * prototypes, -prototypes.square().sum(-1)


def initialize_task(model, weights, support, labels, task_gates, num_classes=None):
    """One initial support forward, with separate prototype and gate branches."""
    features = model.forward_weights(support, weights, embedding=True)
    head = prototype_head(features, labels, num_classes)
    task_embedding = features.mean(dim=0).detach()
    gates = task_gates(task_embedding)
    return head, gates


def update_weights(fast, grads, gates, config):
    """Clip first, gate encoder tensors only, then ordinary SGD on W and b."""
    encoder_count = len(fast) - 2
    if len(grads) != len(fast) or gates.shape != (encoder_count,):
        raise ValueError("Expected one scalar gate per encoder tensor only")
    clip = config["grad_clip"]
    if clip is not None:
        grads = [g.clamp(-clip, clip) for g in grads]
    encoder = [w - config["encoder_lr"] * gate * grad
               for w, gate, grad in zip(fast[:-2], gates, grads[:-2])]
    head = [w - config["classifier_lr"] * grad
            for w, grad in zip(fast[-2:], grads[-2:])]
    return encoder + head


@torch.enable_grad()
def adapt(model, weights, support, labels, config, task_gates, num_classes=None):
    head, gates = initialize_task(model, weights, support, labels,
                                  task_gates, num_classes)
    # Sibling clones make encoder/head independent inner-loop coordinates.
    # Otherwise autograd would also differentiate W0(theta) in the first
    # support update. Both branches still reach theta in the OUTER backward.
    fast = [w.clone() for w in weights] + list(head)
    for _ in range(config["inner_steps"]):
        loss = F.cross_entropy(model.forward_weights(support, fast), labels)
        grads = torch.autograd.grad(loss, fast, create_graph=False,
                                    retain_graph=True)
        # The same graph-connected gates are reused for every inner step.
        # Only fast weights participate in autograd.grad; gates are outer-only.
        fast = update_weights(fast, grads, gates, config)
    return fast


def query_loss(model, fast, query, labels):
    logits = model.forward_weights(query, fast)
    if labels.dtype != torch.long or labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("Query labels must be a nonempty vector of class indices")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("Query labels must use the support class indices")
    return logits, F.cross_entropy(logits, labels)
