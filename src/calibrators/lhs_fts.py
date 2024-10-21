from abc import ABC

import torch
from torch import nn

from data import DictDataset
from llm_models.generic import LLMBundle
from .frequency_ts import TokenFrequencyCalibrator, compute_top_bot_dfs, std_proc
from .lhs_ts import LHSModel, LastHiddenStateCalibrator
from .universal_calibration_models.tiered_models import TieredModel


class LHSFTSModel(LHSModel, TieredModel):
    def __init__(self, llm_bundle: LLMBundle):
        super().__init__(llm_bundle)

        self.top_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, tokens=None):
        # x.shape: [hidden_feature_vecs, num_hidden_features]
        # tokens.shape: [hidden_feature_vecs]

        # First temperature scale using hidden state, then use fts top temp to further scale selected tokens.
        x = self.temp_scale(x)
        if self.top_token_ids is not None:
            x[:, self.top_token_ids] = x[:, self.top_token_ids].div(self.top_temp)

        x = torch.softmax(x, dim=1)
        if tokens is not None:
            confs = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
        else:
            confs = torch.max(x, dim=1).values
        return confs # [confs]


class FrequencyLastHiddenStateCalibrator(LastHiddenStateCalibrator, TokenFrequencyCalibrator, ABC):
    def __init__(self, llm_bundle, loss_fn, score_thresh=0.8):
        LastHiddenStateCalibrator.__init__(self, llm_bundle, loss_fn, LHSFTSModel(llm_bundle))
        TokenFrequencyCalibrator.__init__(self, score_thresh)
        #super().__init__(llm_bundle, LHSFTSModel(llm_bundle), loss_fn, "final_hidden_states")

    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        df_top, _ = compute_top_bot_dfs(calibration_dset, self.llm_bundle, self.metric)
        df_top_modified = df_top[df_top["token_values"] >= self.score_thresh]

        self.top_token_ids = torch.as_tensor(df_top_modified["token_ids"].to_numpy())

        self.calibrator_model.set_tokens(self.top_token_ids)
        assert self.calibrator_model.ready

        LastHiddenStateCalibrator.calibrate(self, calibration_dset, **kwargs)


class FLHS_MSR(FrequencyLastHiddenStateCalibrator):
    def metric(self, mean, std, rfr):
        return mean * std_proc(std) * rfr


class FLHS_M(FrequencyLastHiddenStateCalibrator):
    def metric(self, mean, std, rfr):
        return torch.tensor(mean)


class FLHS_S(FrequencyLastHiddenStateCalibrator):
    def metric(self, mean, std, rfr):
        return std_proc(std)


class FLHS_R(FrequencyLastHiddenStateCalibrator):
    def metric(self, mean, std, rfr):
        return rfr


class FLHS_SR(FrequencyLastHiddenStateCalibrator):
    def metric(self, mean, std, rfr):
        return std_proc(std) * rfr


class FLHS_MR(FrequencyLastHiddenStateCalibrator):
    def metric(self, mean, std, rfr):
        return mean * rfr


class FLHS_MS(FrequencyLastHiddenStateCalibrator):
    def metric(self, mean, std, rfr):
        return torch.tensor(mean * std_proc(std))