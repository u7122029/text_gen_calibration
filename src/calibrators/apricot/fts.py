from data import DictDataset

from .base import APRICOT
from ..frequency_ts import FrequencyTS, FrequencyTSMeanOnly, FrequencyTSNoRFR, FrequencyTSNoStd


class APRICOT_FrequencyTS(FrequencyTS, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FrequencyTS.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FrequencyTSMeanOnly(FrequencyTSMeanOnly, APRICOT_FrequencyTS):
    """
    Uses the mean only metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    pass


class APRICOT_FrequencyTSNoRFR(FrequencyTSNoRFR, APRICOT_FrequencyTS):
    """
    Uses the mean+std metric with no response frequency ratio and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    pass


class APRICOT_FrequencyTSNoStd(FrequencyTSNoStd, APRICOT_FrequencyTS):
    """
    Uses the no std metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    pass