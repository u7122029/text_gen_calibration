from .generic import LogitTokenToConfidenceCalibrator
from .universal_calibration_models import TSModel


class TemperatureScaling(LogitTokenToConfidenceCalibrator):
    def __init__(self, llm_bundle):
        super().__init__(llm_bundle, TSModel())