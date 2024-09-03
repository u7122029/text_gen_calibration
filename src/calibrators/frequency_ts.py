from typing import List

import pandas as pd
import torch

from data import DictDataset
from llm_models import TextGenLLMBundle
from utils import dill_load
from .generic import LogitCalibrator
from .universal_calibration_models import TieredTSModel


def std_proc(std, p=2):
    """
    Standard deviation processor.
    Higher standard deviations are given lower scores (penalised). The penalisation factor is controlled by p.
    @param std:
    @param p:
    @return:
    """
    return torch.abs((2*(std - 0.5)) ** p)


def compute_top_bot_dfs(calibration_dset: DictDataset, llm_bundle: TextGenLLMBundle, metric_func):
    token_confs = {}
    response_occurrences = {}

    for i, item in enumerate(calibration_dset):
        logits = llm_bundle.final_hs_to_logits(item["final_hidden_states"].cuda()).cpu()
        tokens = item["tokens"].long()
        prob_vecs = torch.softmax(logits, dim=1)
        assert len(tokens) == len(prob_vecs)
        confs = torch.take_along_dim(prob_vecs, tokens.unsqueeze(1), dim=1).squeeze(1)
        for token, conf in zip(tokens, confs):
            token = token.item()
            conf = conf.item()
            if token not in token_confs:
                token_confs[token] = []
            token_confs[token].append(conf)

            if token not in response_occurrences:
                response_occurrences[token] = set()
            response_occurrences[token].add(i)

    df_top = {
        "token_ids": [],
        "token_values": [],
        "means": [],
        "stds": [],
        "stds_proc": [],
        "response_props": []
    }
    df_bot = {
        "token_ids": [],
        "token_values": [],
        "means": [],
        "stds": [],
        "stds_proc": [],
        "response_props": []
    }
    #total_tokens = sum([len(v) for _, v in token_confs.items()])

    for k, v in token_confs.items():
        v = torch.Tensor(v)
        n_response_occurrences = len(response_occurrences[k])

        # Don't count tokens that have not occurred in less than 10% of responses
        n_response_proportion = n_response_occurrences / len(calibration_dset)
        if n_response_proportion < 0.2:
            continue

        df_top["token_ids"].append(k)
        df_bot["token_ids"].append(k)

        m = torch.mean(v)
        s = torch.std(v, correction=0 if len(v) == 1 else 1)
        #fv = len(v) / total_tokens

        df_top["token_values"].append(metric_func(m, s, n_response_proportion).item())
        df_bot["token_values"].append(metric_func((1 - m), s, n_response_proportion).item())

        df_top["means"].append(m.item())
        df_bot["means"].append(m.item())

        df_top["stds"].append(s.item())
        df_bot["stds"].append(s.item())

        df_top["stds_proc"].append(std_proc(s).item())
        df_bot["stds_proc"].append(std_proc(s).item())

        df_top["response_props"].append(n_response_proportion)
        df_bot["response_props"].append(n_response_proportion)

    df_top["token_str"] = llm_bundle.tokeniser.batch_decode(df_top["token_ids"])
    df_bot["token_str"] = llm_bundle.tokeniser.batch_decode(df_bot["token_ids"])

    df_top = pd.DataFrame(df_top).sort_values("token_values", ascending=False)
    df_bot = pd.DataFrame(df_bot).sort_values("token_values", ascending=False)

    return df_top, df_bot


class FrequencyTS(LogitCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, score_thresh=0.8, _calibrator_model=None):
        if _calibrator_model is None:
            _calibrator_model = TieredTSModel()

        self.score_thresh = score_thresh
        #self.top_token_values = self.top_token_ids = self.bot_token_values = self.bot_token_ids = None
        self.top_token_ids = None

        super().__init__(llm_bundle, _calibrator_model)

    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        df_top, df_bot = compute_top_bot_dfs(calibration_dset, self.llm_bundle, self.metric)

        df_top = df_top[df_top["token_values"] >= self.score_thresh]
        print(df_top)
        self.top_token_ids = torch.as_tensor(df_top["token_ids"].to_numpy())

        #self.top_token_values = torch.as_tensor(df_top["token_values"].to_numpy())
        #self.bot_token_ids = torch.as_tensor(df_bot["token_ids"].to_numpy())
        #self.bot_token_values = torch.as_tensor(df_bot["token_values"].to_numpy())

        self.calibrator_model.set_tokens(self.top_token_ids)
        assert self.calibrator_model.ready
        super().calibrate(calibration_dset, **kwargs)

    def metric(self, mean, std, response_frequency_ratio):
        sf_std = std_proc(std)
        return mean * sf_std * response_frequency_ratio

    def load(self, filepath):
        d = dill_load(filepath)
        self.score_thresh = d["score_thresh"]
        self.top_token_ids = d["top_token_ids"]
        #self.top_token_values = d["top_token_values"]
        #self.bot_token_ids = d["bot_token_ids"]
        #self.bot_token_values = d["bot_token_values"]

        self.calibrator_model.set_tokens(self.top_token_ids)
        super().load(filepath)

    def save(self, filepath, **kwargs):
        _other_entries = self.get_frequency_dict()
        super().save(filepath, _other_entries)

    def get_frequency_dict(self):
        assert self.score_thresh is not None
        assert self.top_token_ids is not None
        #assert self.bot_token_ids is not None
        #assert self.top_token_values is not None
        #assert self.bot_token_values is not None

        return {
            "score_thresh": self.score_thresh,
            "top_token_ids": self.top_token_ids
        }


"""class FrequencyTSTopOnly(FrequencyTS):
    def __init__(self, llm_bundle, top_k=10):
        super().__init__(llm_bundle, top_k, bot_k=0)"""


"""class FrequencyTSBotOnly(FrequencyTS):
    def __init__(self, llm_bundle, bot_k=10):
        super().__init__(llm_bundle, top_k=0, bot_k=bot_k)"""


class FrequencyTSMeanOnly(FrequencyTS):
    """
    FrequencyTSModel that only considers the mean token confidence. Does not factor in anything else.
    """
    def metric(self, mean, std, response_frequency_ratio):
        return mean


class FrequencyTSNoRFR(FrequencyTS):
    """
    FrequencyTSModel that only considers the mean token confidence and their stds. Does not factor in anything else.
    """
    def metric(self, mean, std, response_frequency_ratio):
        return mean * std_proc(std)


class FrequencyTSNoStd(FrequencyTS):
    """
    FrequencyTSModel without response frequency ratio.
    """
    def metric(self, mean, std, response_frequency_ratio):
        return mean * response_frequency_ratio
