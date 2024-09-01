from abc import ABC, abstractmethod

import torch

from .generic import LogitCalibrator
from .universal_calibration_models import PTSModel


class PTSBase(LogitCalibrator, ABC):
    @abstractmethod
    def __init__(self, llm_bundle, *layer_sizes):
        calib_model = PTSModel(*layer_sizes)
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