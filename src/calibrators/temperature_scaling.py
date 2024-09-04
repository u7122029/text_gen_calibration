from .generic import LogitCalibrator
from .universal_calibration_models import TSModel


class TemperatureScaling(LogitCalibrator):
    def __init__(self, llm_bundle, loss_fn):
        super().__init__(llm_bundle, TSModel(), loss_fn)