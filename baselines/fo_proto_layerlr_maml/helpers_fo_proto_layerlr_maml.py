"""Functional FO-Proto-MAML with learned per-tensor encoder step sizes.

Only update gradients are stopped. The support-derived head stays connected
to the original encoder throughout adaptation and the outer backward pass.
The step sizes are NOT detached: the query loss trains them through the
first-order path  theta_K = theta_0 - sum_s alpha_s * g_s  (g_s constant).
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


@torch.enable_grad()
def adapt(model, weights, support, labels, config, num_classes=None, inner_lrs=None):
    if inner_lrs is None:
        raise ValueError("Learned encoder step sizes are required")
    inner_lrs.validate_encoder(model, config["inner_steps"])
    features = model.forward_weights(support, weights, embedding=True)
    head = prototype_head(features, labels, num_classes)
    # Sibling clones make encoder/head independent inner-loop coordinates.
    # Otherwise autograd would also differentiate W0(theta) in the first
    # support update. Both branches still reach theta in the OUTER backward.
    fast = [w.clone() for w in weights] + list(head)
    encoder_rates = inner_lrs.rates()
    for step in range(config["inner_steps"]):
        loss = F.cross_entropy(model.forward_weights(support, fast), labels)
        grads = torch.autograd.grad(loss, fast, create_graph=False,
                                    retain_graph=True)
        grads = [g.detach() for g in grads]
        clip = config["grad_clip"]
        if clip is not None:
            grads = [g.clamp(-clip, clip) for g in grads]
        # Encoder tensors use their learned step size; the two task-local
        # classifier tensors keep the ordinary fixed FO-Proto-MAML step.
        rates = list(encoder_rates[step].unbind()) + [config["classifier_lr"]] * 2
        fast = [w - lr * g for w, lr, g in zip(fast, rates, grads, strict=True)]
    return fast


def query_loss(model, fast, query, labels):
    logits = model.forward_weights(query, fast)
    if labels.dtype != torch.long or labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("Query labels must be a nonempty vector of class indices")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("Query labels must use the support class indices")
    return logits, F.cross_entropy(logits, labels)
