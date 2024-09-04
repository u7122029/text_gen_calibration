from abc import ABC, abstractmethod

from .generic import LogitCalibrator
from .universal_calibration_models import PTSModel


class PTSBase(LogitCalibrator, ABC):
    @abstractmethod
    def __init__(self, llm_bundle, *layer_sizes, loss_fn=None):
        calib_model = PTSModel(*layer_sizes)
        super().__init__(llm_bundle,
                         calib_model,
                         loss_fn)


class PTS_1L(PTSBase):
    def __init__(self, llm_bundle, loss_fn):
        super().__init__(llm_bundle, llm_bundle.vocab_size(), loss_fn=loss_fn)


class PTS_2L(PTSBase):
    def __init__(self, llm_bundle, loss_fn):
        v = llm_bundle.vocab_size()
        super().__init__(llm_bundle, v, v // 100, loss_fn=loss_fn)


class PTS_3L(PTSBase):
    def __init__(self, llm_bundle, loss_fn):
        v = llm_bundle.vocab_size()
        super().__init__(llm_bundle,
                         v,
                         v // 100,
                         v // 200,
                         loss_fn=loss_fn)