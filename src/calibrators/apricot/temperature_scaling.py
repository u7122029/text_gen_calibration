from data import DictDataset
from .base import APRICOT
from llm_models.generic import LLMBundle
from calibrators.universal_calibration_models import TSModel
from calibrators.generic import LogitCalibrator


class APRICOT_TemperatureScaling(APRICOT, LogitCalibrator):
    """
    Uses the APRICOT method to determine the target confidences for each question, then performs temperature scaling on
    each of the logits corresponding to their responses to match these targets.
    """
    def __init__(self, llm_bundle: LLMBundle, loss_fn):
        APRICOT.__init__(self)
        LogitCalibrator.__init__(self, llm_bundle, TSModel(), label_key="target_confs", loss_fn=loss_fn)

    def calibrate(self, calibration_dset: DictDataset, batch_size=1, epochs=50, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, batch_size)
        calibration_dset["target_confs"] = target_accuracies
        LogitCalibrator.calibrate(self, calibration_dset, batch_size, epochs, **kwargs)

