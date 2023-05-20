import torch
import torch.nn as nn
import torch.optim as optim


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256*256, 1)

    def forward(self, x):
        """
            Args:
                x (Torch.Tensor): Input Tensor. Required Shape is (B, C, H, W)
    
            Returns:
                Torch.Tensor: Output Tensor. Expected Shape is (B, C, H, W)
        """
        x = self.fc(x)
        return x

    def get_optimizer(self):
        """
            Returns:
                torch.optim.[].[]: PyTorch optimizer object.
        """
        algorithm = "Adam"
        optimizer = getattr(optim, algorithm)
        optimizer = optimizer(self.parameters())

        return optimizer

    def get_scheduler(self, optimizer):
        """
            Returns:
                torch.optim.lr_scheduler.[]: PyTorch scheduler object.
        """
        method = "StepLR"
        scheduler = getattr(optim.lr_scheduler, method)
        scheduler = scheduler(optimizer, step_size=1)

        return scheduler


if __name__=="__main__":
    base = Base()
    print(type(base.get_optimizer()))
    print(type(base.get_scheduler(base.get_optimizer())))