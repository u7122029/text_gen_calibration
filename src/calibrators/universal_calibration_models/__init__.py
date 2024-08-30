from typing import Optional

import torch
from torch import nn
from torch.nn.functional import sigmoid

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import DEVICE


class AbsModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)


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
        self.linear = nn.Linear(1, 1)

    def forward(self, x, tokens):
        # x.shape: [logit_vec, vocab size]
        x = torch.softmax(x, dim=1)
        x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1)
        x = sigmoid(self.linear(x))
        return x.flatten()  # [confs]


class PlattScalerConfs(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, tokens):
        # x is a list of logit matrices, each of shape [token response length, vocab size]
        # tokens is a list of token vectors, each of shape [token response length]
        out_confs = []
        for logit_matrix, response_tokens in zip(x, tokens):
            conf_matrix = torch.softmax(logit_matrix, dim=1)
            token_response_confs = torch.take_along_dim(conf_matrix, response_tokens.unsqueeze(1), dim=1)
            out_confs.append(torch.mean(token_response_confs))
        out_confs = torch.Tensor(out_confs).unsqueeze(1).to(out_confs[0].device)
        out_confs = sigmoid(self.linear(out_confs))
        return out_confs.flatten()  # [calibrated_confs]


class TieredTSModel(nn.Module):
    """
    Contains 3 temperature parameters.
    One determines the adjustment of the token ids that commonly occur with high confidence
    One determines the adjustment of the token ids that commonly occur with low confidence
    The last is a general temperature that adjusts all the tokens after adjustment from the previous two temps.
    """

    def __init__(self):
        super().__init__()
        self.top_token_ids = None
        self.bot_token_ids = None

        self.top_temp = nn.Parameter(torch.tensor(1.0))
        self.bot_temp = nn.Parameter(torch.tensor(1.0))
        self.general_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, tokens=None):
        # x.shape: [logit_vec, vocab size]
        if self.top_token_ids is not None:
            x[:, self.top_token_ids] = x[:, self.top_token_ids] / self.top_temp

        if self.bot_token_ids is not None:
            x[:, self.bot_token_ids] = x[:, self.bot_token_ids] / self.bot_temp

        x = x / self.general_temp
        x = torch.softmax(x, dim=1)
        if tokens is not None:
            x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
        else:
            x = torch.max(x, dim=1).values
        return x  # [confs]

    def set_tokens(self, top_token_ids: Optional[torch.Tensor], bot_token_ids: Optional[torch.Tensor]):
        self.top_token_ids = top_token_ids
        self.bot_token_ids = bot_token_ids


class PTSModel(nn.Module):
    def __init__(self, *layer_sizes):
        """
        Constructor for a generic Parametric Temperature Scaling Model.
        :param layer_sizes:
        """
        assert len(layer_sizes) >= 1
        assert all([isinstance(x, int) for x in layer_sizes])
        super().__init__()

        # Build model
        self.first_layer_size = layer_sizes[0]
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=layer_sizes[-1], out_features=1))
        self.layers.append(AbsModule())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inp, tokens):
        t, _ = torch.sort(torch.topk(inp, self.first_layer_size, dim=1).values, dim=1, descending=True)
        t = torch.clip(self.layers(t), min=1e-8, max=1e+8)

        x = inp / t
        x = torch.softmax(x, dim=1)
        x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)

        return x


class TieredPTSModel(nn.Module):
    """
    TODO: FINISH THIS MODEL!
    """

    def __init__(self):
        super().__init__()
        self.top_token_ids = None
        self.bot_token_ids = None

        self.top_linear = None
        self.bot_temp = nn.Parameter(torch.tensor(1.0))
        self.general_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, tokens=None):
        # x.shape: [logit_vec, vocab size]
        if self.top_token_ids is not None:
            x[:, self.top_token_ids] = x[:, self.top_token_ids] / self.top_temp

        if self.bot_token_ids is not None:
            x[:, self.bot_token_ids] = x[:, self.bot_token_ids] / self.bot_temp

        x = x / self.general_temp
        x = torch.softmax(x, dim=1)
        if tokens is not None:
            x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
        else:
            x = torch.max(x, dim=1).values
        return x  # [confs]

    def set_tokens(self, top_token_ids: Optional[torch.Tensor], bot_token_ids: Optional[torch.Tensor]):
        self.top_token_ids = top_token_ids
        self.bot_token_ids = bot_token_ids


class TokenCalibratorModel(nn.Module):
    """
    Uses a sequence classification model that takes a question + its response, then outputs the calibrated confidence.
    """

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")
        self.tokeniser = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def forward(self, x):
        # x is a list of string inputs.
        x = self.tokeniser(x, return_tensors="pt", padding=True).to(self.device)
        x = self.model(**x)
        x = torch.softmax(x.logits, dim=-1)[:, 1]
        return x  # [confs]
