from .base import APRICOT
from ..lhs_fts import *


class APRICOT_FLHS_MSR(FLHS_MSR, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, validation_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FLHS_MSR.calibrate(self, calibration_dset, validation_dset, **kwargs)


class APRICOT_FLHS_M(FLHS_M, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, validation_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FLHS_M.calibrate(self, calibration_dset, validation_dset, **kwargs)


class APRICOT_FLHS_S(FLHS_S, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, validation_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FLHS_S.calibrate(self, calibration_dset, validation_dset, **kwargs)


class APRICOT_FLHS_R(FLHS_R, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, validation_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FLHS_R.calibrate(self, calibration_dset, validation_dset, **kwargs)


class APRICOT_FLHS_MS(FLHS_MS, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, validation_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FLHS_MS.calibrate(self, calibration_dset, validation_dset, **kwargs)


class APRICOT_FLHS_MR(FLHS_MR, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, validation_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FLHS_MR.calibrate(self, calibration_dset, validation_dset, **kwargs)


class APRICOT_FLHS_SR(FLHS_SR, APRICOT):
    def calibrate(self, calibration_dset: DictDataset, validation_dset: DictDataset, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        FLHS_SR.calibrate(self, calibration_dset, validation_dset, **kwargs)