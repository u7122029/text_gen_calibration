from data import DictDataset
from llm_models import LLMBundle
from .base import APRICOT
from ..lhs_ts import LastHiddenStateCalibrator


class APRICOT_LHS(APRICOT, LastHiddenStateCalibrator):
    def __init__(self, llm_bundle: LLMBundle, loss_fn):
        APRICOT.__init__(self)
        LastHiddenStateCalibrator.__init__(self, llm_bundle, loss_fn)

    def calibrate(self, calibration_dset: DictDataset, batch_size=1, epochs=35, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, batch_size)
        calibration_dset["target_confs"] = target_accuracies
        LastHiddenStateCalibrator.calibrate(self, calibration_dset, batch_size, epochs, **kwargs)
