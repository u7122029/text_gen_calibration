from data import DictDataset

from .base import APRICOT
from ..frequency_ts import *


class APRICOT_FrequencyTS_MSR(FrequencyTS_MSR, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FrequencyTS_MSR.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FrequencyTS_M(FrequencyTS_M, APRICOT_FrequencyTS_MSR):
    """
    Uses the mean only metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    pass


class APRICOT_FrequencyTS_MS(FrequencyTS_MS, APRICOT_FrequencyTS_MSR):
    """
    Uses the mean+std metric with no response frequency ratio and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    pass


class APRICOT_FrequencyTS_MR(FrequencyTS_MR, APRICOT_FrequencyTS_MSR):
    """
    Uses the no std metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    pass


class APRICOT_FrequencyTS_R(FrequencyTS_R, APRICOT_FrequencyTS_MSR):
    """
    Only uses the relative response frequency metric.
    Nothing further needs to be defined here.
    """
    pass