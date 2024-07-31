from .base import APRICOT
from utils import LLMBundle
from calibrators.universal_calibration_models import TSModel
from calibrators.generic import LogitTokenToConfidenceCalibrator


class APRICOT_TemperatureScaling(APRICOT, LogitTokenToConfidenceCalibrator):
    """
    Uses the APRICOT method to determine the target confidences for each question, then performs temperature scaling on
    each of the logits corresponding to their responses to match these targets.
    """
    def __init__(self, llm_bundle: LLMBundle):
        APRICOT.__init__(self, llm_bundle)
        LogitTokenToConfidenceCalibrator.__init__(self, llm_bundle, TSModel(), "target_confs")



