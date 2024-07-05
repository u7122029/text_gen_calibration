from .generic import LogitTokenToConfidenceCalibrator
from abc import ABC, abstractmethod
from torch import nn
from utils import AbsModule
import torch


class PTSBase(LogitTokenToConfidenceCalibrator, ABC):
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

        def forward(self, inp, tokens=None):
            t, _ = torch.sort(torch.topk(inp, self.first_layer_size, dim=1).values, dim=1, descending=True)
            t = torch.clip(self.layers(t), min=1e-8, max=1e+8)

            x = inp / t
            x = torch.softmax(x, dim=1)

            if tokens is not None:
                x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
            else:
                x = torch.max(x, dim=1).values

            return x

    @abstractmethod
    def __init__(self, llm_bundle, *layer_sizes):
        calib_model = PTSBase.PTSModel(*layer_sizes)
        print(calib_model)
        super().__init__(llm_bundle,
                         calib_model)


class PTS_1L(PTSBase):
    def __init__(self, llm_bundle):
        super().__init__(llm_bundle, llm_bundle.vocab_size())


class PTS_2L(PTSBase):
    def __init__(self, llm_bundle):
        v = llm_bundle.vocab_size()
        super().__init__(llm_bundle, v, v // 100)


class PTS_3L(PTSBase):
    def __init__(self, llm_bundle):
        v = llm_bundle.vocab_size()
        super().__init__(llm_bundle,
                         v,
                         v // 100,
                         v // 200)