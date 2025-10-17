
import torch

class ReactivationBurst:
    def __init__(self, model, recovery_rate=0.1):
        self.model = model
        self.recovery_rate = recovery_rate

    def reactivate(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                mask = (torch.rand_like(param) < self.recovery_rate).float()
                param.data.add_(mask * torch.randn_like(param) * 0.01)
