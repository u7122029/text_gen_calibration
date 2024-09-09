from .frequency_ts import FrequencyTS, compute_top_bot_dfs, std_proc
from .universal_calibration_models.tiered_models import TieredScalerModel


class FrequencyScaler(FrequencyTS):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, loss_fn, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredScalerModel())
