import torch
import torch.nn as nn
from typing import List
    
    
def update_weights(weights: List[torch.Tensor], 
                   grads: List[torch.Tensor], 
                   grad_clip: int, 
                   lr: float,
                   gate_logits: List[torch.Tensor],
                   weight_names,
                   low_rank_transport,
                   diagnostics=None) -> List[torch.Tensor]:
    """ Apply scalar gates and an additive output-channel Conv2d correction.

    Args:
        weights (List[torch.Tensor]): Weights to be updated.
        grads (List[torch.Tensor]): Gradients for each weight.
        grad_clip (int): Boundary for clipping the gradients.
        lr (float): Learning rate.
        gate_logits (List[torch.Tensor]): Static scalar meta-parameters. These
            are never part of the support-gradient inputs or updated here.
        weight_names: Actual parameter names in the fast-weight list's order.
        low_rank_transport: Static U/V meta-parameters for Conv2d weights only.
        diagnostics: Optional detached observer for existing validation steps.

    Returns:
        List[torch.Tensor]: Updated weights
    """
    if not (len(weights) == len(grads) == len(gate_logits)):
        raise ValueError("Each fast-weight tensor must have one scalar gate.")
    if any(gate.ndim != 0 for gate in gate_logits):
        raise ValueError("Gate logits must be scalar tensors.")
    low_rank_transport.validate_fast_weights(weight_names, weights)

    new_weights = []
    for name, weight, grad, logit in zip(weight_names, weights, grads, gate_logits):
        if grad is None:
            new_weights.append(weight)
            continue
        # Same clipping and scalar-update evaluation order as SGMAML. Keeping
        # the additive correction separate also preserves U=0 starting parity.
        if grad_clip is not None:
            grad = torch.clamp(grad, -grad_clip, +grad_clip)
        gate = torch.sigmoid(logit)
        updated = weight - lr * gate * grad
        correction = low_rank_transport.correction(name, grad)
        if correction is not None:
            # No gate on R and no post-transform clipping/normalization.
            updated = updated - lr * correction
            if diagnostics is not None:
                with torch.no_grad():
                    scalar_part = gate.detach() * grad.detach()
                    diagnostics.record(name, grad.detach(), scalar_part,
                                       correction.detach(),
                                       scalar_part + correction.detach())
        new_weights.append(updated)
    
    return new_weights


def get_grads(model: nn.Module, 
              X_train: torch.Tensor, 
              y_train: torch.Tensor, 
              weights: List[torch.Tensor] = None, 
              second_order: bool = False, 
              retain_graph: bool = False) -> List[torch.Tensor]:
    """ Compute the gradients of processing the specified input.

    Args:
        model (nn.Module): Model to be used.
        X_train (torch.Tensor): Support set images.
        y_train (torch.Tensor): Support set labels.
        weights (List[torch.Tensor], optional): Weights to be used by the 
            model. Defaults to None.
        second_order (bool, optional): Boolean flag to control if second order
            derivatives should be computed. Defaults to False.
        retain_graph (bool, optional): Boolean flag to control if the 
            derivation graph should be retained. Defaults to False.

    Returns:
        List[torch.Tensor]: Gradients of the operation
    """
    model.zero_grad()
    if weights is None:
        weights = model.parameters()
        out = model(X_train)
    else:
        out = model.forward_weights(X_train, weights)
    
    loss = model.criterion(out, y_train)
    grads = torch.autograd.grad(loss, weights, create_graph=second_order, 
        retain_graph=retain_graph)
    
    return list(grads)
    

def process_support_set(model: nn.Module, 
                        weights: List[torch.Tensor], 
                        X_train: torch.Tensor, 
                        y_train: torch.Tensor, 
                        num_classes: int) -> torch.Tensor:
    """ Process the support set following the Prototypical Networks strategy.

    Args:
        model (nn.Module): Model to be used.
        weights (List[torch.Tensor]): Weights to be used by the model.
        X_train (torch.Tensor): Support set images.
        y_train (torch.Tensor): Support set labels.
        num_classes (int): Number of classes to predict.

    Returns:
        torch.Tensor: Support prototypes.
    """
    # Compute input embeddings
    support_embeddings = model.forward_weights(X_train, weights, 
        embedding=True)
    
    # Compute prototypes
    prototypes = torch.zeros((num_classes, support_embeddings.size(1)), 
        device=weights[0].device)
    for i in range(num_classes):
        mask = y_train == i
        prototypes[i] = (support_embeddings[mask].sum(dim=0) / 
            torch.sum(mask).item())
        
    return prototypes


def process_query_set(model: nn.Module, 
                      weights: List[torch.Tensor], 
                      X_test: torch.Tensor, 
                      prototypes: torch.Tensor) -> torch.Tensor:
    """ Process the query set following the Matching Networks strategy.

    Args:
        model (nn.Module): Model to be used.
        weights (List[torch.Tensor]): Weights to be used by the model.
        X_test (torch.Tensor): Query set images.
        prototypes (torch.Tensor): Support prototypes

    Returns:
        torch.Tensor: Distances to prototypes.
    """
    # Compute input embeddings
    query_embeddings = model.forward_weights(X_test, weights, embedding=True)

    # Create distance matrix (negative predictions)
    distance_matrix = (torch.cdist(query_embeddings.unsqueeze(0), 
        prototypes.unsqueeze(0))**2).squeeze(0) 
    out = -1 * distance_matrix
    
    return out


def get_grads_ncc(model: nn.Module, 
                  X_train: torch.Tensor, 
                  y_train: torch.Tensor, 
                  X_test: torch.Tensor, 
                  y_test: torch.Tensor, 
                  num_classes: int,
                  weights: List[torch.Tensor] = None, 
                  second_order: bool = False, 
                  retain_graph: bool = False) -> List[torch.Tensor]:
    """ Compute the gradients of processing the specified input.

    Args:
        model (nn.Module): Model to be used.
        X_train (torch.Tensor): Support set images.
        y_train (torch.Tensor): Support set labels.
        X_test (torch.Tensor): Query set images.
        y_test (torch.Tensor): Query set labels.
        num_classes (int): Number of classes to predict.
        weights (List[torch.Tensor], optional): Weights to be used by the 
            model. Defaults to None.
        second_order (bool, optional): Boolean flag to control if second order
            derivatives should be computed. Defaults to False.
        retain_graph (bool, optional): Boolean flag to control if the 
            derivation graph should be retained. Defaults to False.

    Returns:
        List[torch.Tensor]: Gradients of the operation
    """
    model.zero_grad()
    if weights is None:
        weights = model.parameters()
    
    prototypes = process_support_set(model, weights, X_train, y_train, 
        num_classes)
    out = process_query_set(model, weights, X_test, prototypes)
    
    loss = model.criterion(out, y_test) / (num_classes * len(y_test)) 
    grads = torch.autograd.grad(loss, weights, create_graph=second_order, 
        retain_graph=retain_graph, allow_unused=True)
    
    return list(grads)
