from .generic import LogitTokenToConfidenceCalibrator
from .universal_calibration_models import PlattScalerModel


class PlattScaling(LogitTokenToConfidenceCalibrator):
    def __init__(self, llm_bundle):
        super().__init__(llm_bundle, PlattScalerModel())



