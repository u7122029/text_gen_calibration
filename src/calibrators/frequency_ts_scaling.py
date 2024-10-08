from .frequency_ts import *
from .universal_calibration_models.tiered_models import TieredScalerModel, TieredPlattModel


class FTP(LogitTokenFrequencyCalibrator):
    """
    Frequency Temperature Platt
    Performs temperature scaling on all logits, then performs platt scaling on the high xi tokens.
    """
    def __init__(self, llm_bundle, loss_fn, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredScalerModel())


class FTP_M(FTP):
    def metric(self, mean, std, rfr):
        return mean


class FTP_S(FTP):
    def metric(self, mean, std, rfr):
        return std_proc(std)


class FTP_R(FTP):
    def metric(self, mean, std, rfr):
        return rfr


class FTP_MR(FTP):
    def metric(self, mean, std, rfr):
        return mean * rfr


class FTP_MS(FTP):
    def metric(self, mean, std, rfr):
        return mean * std_proc(std)


class FTP_SR(FTP):
    def metric(self, mean, std, rfr):
        return std_proc(std) * rfr


class FTP_MSR(FTP):
    def metric(self, mean, std, rfr):
        return mean * std_proc(std) * rfr


class FPS(LogitTokenFrequencyCalibrator):
    """
    Frequency Platt Scaling
    Performs platt scaling on all logits, then performs platt scaling on high xi tokens.
    """
    def __init__(self, llm_bundle, loss_fn, score_thresh=0.8):
        super().__init__(llm_bundle, loss_fn, score_thresh, TieredPlattModel())


class FPS_M(FPS):
    def metric(self, mean, std, rfr):
        return mean


class FPS_S(FPS):
    def metric(self, mean, std, rfr):
        return std_proc(std)


class FPS_R(FPS):
    def metric(self, mean, std, rfr):
        return rfr


class FPS_MR(FPS):
    def metric(self, mean, std, rfr):
        return mean * rfr


class FPS_MS(FPS):
    def metric(self, mean, std, rfr):
        return mean * std_proc(std)


class FPS_SR(FPS):
    def metric(self, mean, std, rfr):
        return std_proc(std) * rfr


class FPS_MSR(FPS):
    def metric(self, mean, std, rfr):
        return mean * std_proc(std) * rfr