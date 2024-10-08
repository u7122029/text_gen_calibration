from .base import APRICOT
from ..frequency_ts_scaling import *


class APRICOT_FTP_MSR(FTP_MSR, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FTP_MSR.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FTP_M(FTP_M, APRICOT):
    """
    Uses the mean only metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """

    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FTP_M.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FTP_S(FTP_S, APRICOT):
    """
    Uses the mean only metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """

    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FTP_S.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FTP_MS(FTP_MS, APRICOT):
    """
    Uses the mean+std metric with no response frequency ratio and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FTP_MS.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FTP_MR(FTP_MR, APRICOT):
    """
    Uses the no std metric and the calibrate function from APRICOT_FrequencyTS.
    Nothing further needs to be defined here.
    """
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FTP_MR.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FTP_R(FTP_R, APRICOT):
    """
    Only uses the relative response frequency metric.
    Nothing further needs to be defined here.
    """
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FTP_R.calibrate(self, calibration_dset, **kwargs)


class APRICOT_FTP_SR(FTP_SR, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FTP_R.calibrate(self, calibration_dset, **kwargs)