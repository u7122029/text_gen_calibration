from calibrators.frequency_ts import FrequencyTS_MSR
from calibrators.universal_calibration_models import TieredPTSModel


class FrequencyPTS(FrequencyTS_MSR):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, loss_fn=None, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPTSModel())

