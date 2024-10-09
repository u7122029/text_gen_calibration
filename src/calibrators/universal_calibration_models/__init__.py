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
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, tokens):
        # x.shape: [logit_vec, vocab size]
        x.div_(self.temperature)
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
        x = torch.sigmoid_(self.linear(x))
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
        t = torch.clip_(nn.functional.softplus(self.layers(t)), min=1e-5)

        inp.div_(inp, t)
        inp = torch.softmax(inp, dim=1)
        inp = torch.take_along_dim(inp, tokens.unsqueeze(1), dim=1).squeeze(1)

        return inp


class TokenCalibratorModel(nn.Module):
    """
    Uses a sequence classification model that takes a question + its response, then outputs the calibrated confidence.
    """

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")
        self.tokeniser = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.to(device)

    def forward(self, x):
        # x is a list of string inputs.
        x = self.tokeniser(x, return_tensors="pt", padding=True).to(self.device)
        x = self.model(**x)
        x = torch.softmax(x.logits, dim=-1)[:, 1]
        return x  # [confs]
