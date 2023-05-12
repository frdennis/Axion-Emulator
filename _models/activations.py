import torch
import torch.nn as nn


def speculator_activation(x, gamma, beta):
    """
    https://arxiv.org/pdf/1911.11778.pdf (Eq 8)
    """
    return (gamma + torch.sigmoid(beta * x) * (1.0 - gamma)) * x


class Speculator(nn.Module):
    """
    """

    def __init__(self, dim_input):
        """
        """
        super().__init__()
        self.gamma = nn.Parameter(nn.init.normal_(torch.empty(dim_input)))
        self.beta = nn.Parameter(nn.init.normal_(torch.empty(dim_input)))

    def forward(self, x):
        """
        """
        return speculator_activation(x, gamma=self.gamma, beta=self.beta)
