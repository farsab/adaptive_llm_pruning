
import torch

class AdaptivePruner:
    def __init__(self, model, target_sparsity=0.4, alpha=0.9):
        self.model = model
        self.target_sparsity = target_sparsity
        self.alpha = alpha
        self.masks = {}

    def update_masks(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                grad_var = param.grad.var().item() if param.grad is not None else 0.0
                threshold = grad_var * self.alpha
                mask = (param.abs() > threshold).float()
                self.masks[name] = mask
                param.data.mul_(mask)
