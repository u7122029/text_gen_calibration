import torch

from data import DictDataset

from .base import APRICOT
from ..frequency_ts import FrequencyTS


class APRICOT_FrequencyTS(APRICOT, FrequencyTS):
    def calibrate(self, calibration_dset: DictDataset, top_k=10, bot_k=10, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, kwargs["batch_size"])
        calibration_dset["target_confs"] = target_accuracies
        super().calibrate(calibration_dset, top_k, bot_k, **kwargs)


class APRICOT_FrequencyTSTopOnly(APRICOT_FrequencyTS):
    def calibrate(self, calibration_dset: DictDataset, top_k=10, **kwargs):
        kwargs["bot_k"] = 0
        super().calibrate(calibration_dset, top_k, **kwargs)


class APRICOT_FrequencyTSBotOnly(APRICOT_FrequencyTS):
    def calibrate(self, calibration_dset: DictDataset, bot_k=10, **kwargs):
        kwargs["top_k"] = 0
        super().calibrate(calibration_dset, bot_k=bot_k, **kwargs)


class APRICOT_FrequencyTSMeanOnly(APRICOT_FrequencyTS):
    """
    FrequencyTSModel that only considers the mean token confidence. Does not factor in anything else.
    """
    def __compute_metric(self, mean, std, token_frequency, response_frequency_ratio):
        return mean


class APRICOT_FrequencyTSMeanStdOnly(APRICOT_FrequencyTS):
    """
    FrequencyTSModel that only considers the mean token confidence and their stds. Does not factor in anything else.
    """
    def __compute_metric(self, mean, std, token_frequency, response_frequency_ratio):
        sf = lambda x: -2 * x + 1
        return mean * sf(std)


class APRICOT_FrequencyTSNoRF(APRICOT_FrequencyTS):
    """
    FrequencyTSModel without response frequency ratio.
    """
    def __compute_metric(self, mean, std, token_frequency, response_frequency_ratio):
        sf = lambda x: -2 * x + 1
        f = lambda x: -1 / (x / 4 + 1) + 1
        return mean * sf(std) * f(token_frequency)


class APRICOT_FrequencyTSNoTF(FrequencyTS):
    """
    FrequencyTSModel without token frequency.
    """
    def __compute_metric(self, mean, std, token_frequency, response_frequency_ratio):
        sf = lambda x: torch.abs((2*(x - 0.5)) ** 16)
        return mean * sf(std) * (response_frequency_ratio ** 10)