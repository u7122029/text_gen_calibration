import torch

from calibrators.frequency_ts import LogitTokenFrequencyCalibrator, std_proc
from calibrators.universal_calibration_models.tiered_models import TieredPTSModel


class FrequencyPTS_MSR(LogitTokenFrequencyCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, loss_fn=None, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPTSModel())

    def metric(self, mean, std, response_frequency_ratio):
        return mean * std_proc(std) * response_frequency_ratio


class FrequencyPTS_M(LogitTokenFrequencyCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, loss_fn=None, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPTSModel())

    def metric(self, mean, std, response_frequency_ratio):
        return torch.tensor(mean)


class FrequencyPTS_S(LogitTokenFrequencyCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, loss_fn=None, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPTSModel())

    def metric(self, mean, std, response_frequency_ratio):
        return std_proc(std)


class FrequencyPTS_R(LogitTokenFrequencyCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, loss_fn=None, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPTSModel())

    def metric(self, mean, std, response_frequency_ratio):
        return torch.tensor(response_frequency_ratio)


class FrequencyPTS_MS(LogitTokenFrequencyCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, loss_fn=None, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPTSModel())

    def metric(self, mean, std, response_frequency_ratio):
        return mean * std_proc(std)


class FrequencyPTS_SR(LogitTokenFrequencyCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, loss_fn=None, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPTSModel())

    def metric(self, mean, std, response_frequency_ratio):
        return torch.tensor(std_proc(std) * response_frequency_ratio)


class FrequencyPTS_MR(LogitTokenFrequencyCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, loss_fn=None, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPTSModel())

    def metric(self, mean, std, response_frequency_ratio):
        return torch.tensor(mean * response_frequency_ratio)
