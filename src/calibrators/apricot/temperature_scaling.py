from data import DictDataset
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

    def calibrate(self, calibration_dset: DictDataset, batch_size=1, epochs=30, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, batch_size)
        calibration_dset.add_column("target_confs", target_accuracies)
        LogitTokenToConfidenceCalibrator.calibrate(self, calibration_dset, batch_size, epochs, **kwargs)

