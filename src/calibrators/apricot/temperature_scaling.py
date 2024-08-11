from data import DictDataset
from data.folderdataset import FolderDataset
from .base import APRICOT
from llm_models import LLMBundle
from calibrators.universal_calibration_models import TSModel
from calibrators.generic import LogitCalibrator


class APRICOT_TemperatureScaling(APRICOT, LogitCalibrator):
    """
    Uses the APRICOT method to determine the target confidences for each question, then performs temperature scaling on
    each of the logits corresponding to their responses to match these targets.
    """
    def __init__(self, llm_bundle: LLMBundle):
        APRICOT.__init__(self, llm_bundle)
        LogitCalibrator.__init__(self, llm_bundle, TSModel(), "target_confs")

    def calibrate(self, calibration_dset: FolderDataset, batch_size=1, epochs=30, **kwargs):
        embeddings, target_confs = self.get_target_accuracies(calibration_dset, batch_size)
        calibration_dset = calibration_dset.to_dictdataset()
        calibration_dset["target_confs"] = target_confs
        LogitCalibrator.calibrate(self, calibration_dset, batch_size, epochs, **kwargs)

