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


class PlattScalerLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x, tokens):
        # x.shape: [logit_vec, vocab size]
        x = torch.softmax(x, dim=1)
        x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1)
        x = sigmoid(self.linear(x))
        return x.flatten()  # [confs]


class PlattScalerConfs(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        # x.shape: [response_no (batch_size), confidence (1)]
        x = sigmoid(self.linear(x))
        return x.flatten()  # [calibrated_confs]