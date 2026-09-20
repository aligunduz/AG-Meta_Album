import torch
import torch.nn as nn
from typing import List
    
    
def update_weights(weights: List[torch.Tensor], 
                   grads: List[torch.Tensor], 
                   grad_clip: int, 
                   lr: float,
                   gate_logits: List[torch.Tensor]) -> List[torch.Tensor]:
    """ Apply one shared, learned scalar gate per fast-weight tensor.

    Args:
        weights (List[torch.Tensor]): Weights to be updated.
        grads (List[torch.Tensor]): Gradients for each weight.
        grad_clip (int): Boundary for clipping the gradients.
        lr (float): Learning rate.
        gate_logits (List[torch.Tensor]): Static scalar meta-parameters. These
            are never part of the support-gradient inputs or updated here.

    Returns:
        List[torch.Tensor]: Updated weights
    """
    if not (len(weights) == len(grads) == len(gate_logits)):
        raise ValueError("Each fast-weight tensor must have one scalar gate.")
    if any(gate.ndim != 0 for gate in gate_logits):
        raise ValueError("Gate logits must be scalar tensors.")

    if grad_clip is not None:
        grads = [torch.clamp(p, -grad_clip, +grad_clip) if p is not None else 0 
                 for p in grads]
    
    # Preserve MAML's clipping; only scale the resulting support gradient.
    # First-order support gradients have no graph, but sigmoid must retain
    # its graph so that query loss can train the gate logits.
    new_weights = [weight if grad is None else
                   weight - lr * torch.sigmoid(gate) * grad
                   for weight, grad, gate in zip(weights, grads, gate_logits)]
    
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
