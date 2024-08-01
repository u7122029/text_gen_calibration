import torch
from torch import nn
from torch.nn.functional import sigmoid


class TSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, tokens):
        # x.shape: [logit_vec, vocab size]
        x = x / self.temperature
        x = torch.softmax(x, dim=1)
        x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)

        return x  # [confs]


class PlattScalerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x, tokens):
        # x.shape: [logit_vec, vocab size]
        x = torch.softmax(x, dim=1)
        x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1)
        x = sigmoid(self.linear(x))
        return x.flatten()  # [confs]