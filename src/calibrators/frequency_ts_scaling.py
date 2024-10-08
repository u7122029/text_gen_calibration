from .frequency_ts import *
from .universal_calibration_models.tiered_models import TieredScalerModel, TieredPlattModel


class FTP(LogitTokenFrequencyCalibrator):
    """
    Frequency Temperature Platt
    Performs temperature scaling on all logits, then performs platt scaling on the high xi tokens.
    """
    def __init__(self, llm_bundle, loss_fn, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredScalerModel())


class FPS(LogitTokenFrequencyCalibrator):
    """
    Frequency Platt Scaling
    Performs platt scaling on all logits, then performs platt scaling on high xi tokens.
    """
    def __init__(self, llm_bundle, loss_fn, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPlattModel())
