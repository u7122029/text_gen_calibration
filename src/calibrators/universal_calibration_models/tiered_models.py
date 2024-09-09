from typing import Optional
from abc import ABC, abstractmethod
import torch
from torch import nn


class PhaseModel(nn.Module):
    """
    NOTE: may not need this class.
    A Tiered Model contains two phases of training.
    The first phase optimises the temperature scaling parameter while freezing the other parameters.
    The second phases optmises the remaining parameters while freezing the temperature scaling parameter.
    """
    def __init__(self, model1: nn.Module, model2: nn.Module):
        """
        PhaseModel Constructor
        @param model1: The phase 1 model - must accept tokens along with logit input
        @param model2: The phase 2 model - must accept tokens along with logit input
        """
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.__phase = 0

    @property
    def phase(self):
        return self.__phase

    @phase.setter
    def phase(self, new_phase):
        assert new_phase in [1, 2]
        if new_phase == 1:
            # Unfreeze model1 parameters
            for param in self.model1.parameters():
                param.requires_grad = True

            # Freeze model2 parameters
            for param in self.model2.parameters():
                param.requires_grad = False
        else:
            # Freeze model1 parameters
            for param in self.model2.parameters():
                param.requires_grad = False

            # Unfreeze model2 parameters
            for param in self.model1.parameters():
                param.requires_grad = True

    def forward(self, x, tokens=None):
        return self.model2(self.model1(x, tokens), tokens)


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
        #self.bot_token_ids = None

        self.top_temp = nn.Parameter(torch.tensor(1.0))
        self.general_temp = nn.Parameter(torch.tensor(1.0))

        self.ready = False

    def forward(self, x, tokens=None):
        # x.shape: [logit_vec, vocab size]
        x = x / self.general_temp

        if self.top_token_ids is not None:
            x[:, self.top_token_ids] = x[:, self.top_token_ids] / self.top_temp

        x = torch.softmax(x, dim=1)
        if tokens is not None:
            x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
        else:
            x = torch.max(x, dim=1).values
        return x  # [confs]

    def set_tokens(self, top_token_ids: Optional[torch.Tensor]):
        self.top_token_ids = top_token_ids
        self.ready = True


class TieredScalerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_token_ids = None
        #self.bot_token_ids = None

        self.a = nn.Parameter(torch.tensor(-4.0))
        self.b = nn.Parameter(torch.tensor(-0.5))
        self.general_temp = nn.Parameter(torch.tensor(1.0))

        self.ready = False

    def forward(self, x, tokens=None):
        # x.shape: [logit_vec, vocab size]
        x = x / self.general_temp

        x = torch.softmax(x, dim=1)
        if tokens is not None:
            x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
        else:
            x = torch.max(x, dim=1).values

        if self.top_token_ids is not None and tokens is not None:
            mask = torch.isin(tokens, self.top_token_ids.to(tokens.device))
            x[mask] = 1 / (1 + torch.exp(self.a * x[mask] + self.b))

        return x  # [confs]

    def set_tokens(self, top_token_ids: Optional[torch.Tensor]):
        self.top_token_ids = top_token_ids
        self.ready = True
