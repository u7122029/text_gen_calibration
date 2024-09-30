from data import DictDataset
from llm_models.generic import LLMBundle
from .base import APRICOT
from ..token_response_scaler import TokenCalibrator


class APRICOT_Original(APRICOT, TokenCalibrator):
    """
    Uses the APRICOT method to determine the target confidences for each question in the calibration dataset.
    Then we train an LLM for sequence classification to ensure that each tokenised question with response attains a
    confidence that is as close as possible to these targets.
    This method of optimisation corresponds with the original APRICOT method proposed in the respective paper.
    """
    def __init__(self, llm_bundle: LLMBundle, loss_fn):
        APRICOT.__init__(self)
        TokenCalibrator.__init__(self, llm_bundle, loss_fn, label_key="target_confs")

    def calibrate(self, calibration_dset: DictDataset, validation_dset: DictDataset, batch_size=1, epochs=35, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, batch_size)
        calibration_dset["target_confs"] = target_accuracies

        TokenCalibrator.calibrate(self, calibration_dset, validation_dset, batch_size, epochs, **kwargs)
