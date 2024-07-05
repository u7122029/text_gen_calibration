from .generic import LogitTokenToConfidenceCalibrator
from torch import nn
import torch


class TemperatureScalingVariant(LogitTokenToConfidenceCalibrator):
    class TSModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, x, tokens=None):
            # x.shape: [logit_vec, vocab size]
            x = x / self.temperature
            x = torch.softmax(x, dim=1)
            if tokens is not None:
                x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
            else:
                x = torch.max(x, dim=1).values
            return x  # [confs]

    def __init__(self, llm_bundle):
        super().__init__(llm_bundle, TemperatureScalingVariant.TSModel())