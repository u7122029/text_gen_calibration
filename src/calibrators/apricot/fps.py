from .base import APRICOT
from ..frequency_ts_scaling import *


class APRICOT_FPS_MSR(FPS_MSR, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FPS_MSR.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FPS_M(FPS_M, APRICOT):
    """
    Uses the mean only metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """

    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FPS_M.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FPS_S(FPS_S, APRICOT):
    """
    Uses the mean only metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """

    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FPS_S.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FPS_MS(FPS_MS, APRICOT):
    """
    Uses the mean+std metric with no response frequency ratio and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FPS_MS.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FPS_MR(FPS_MR, APRICOT):
    """
    Uses the no std metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FPS_MR.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FPS_R(FPS_R, APRICOT):
    """
    Only uses the relative response frequency metric.
    Nothing further needs to be defined here.
    """
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FPS_R.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FPS_SR(FPS_SR, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FPS_R.calibrate(self, calibration_dset, **kwargs)