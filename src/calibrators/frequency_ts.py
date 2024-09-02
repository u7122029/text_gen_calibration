from typing import List

import pandas as pd
import torch

from data import DictDataset
from llm_models import TextGenLLMBundle
from utils import dill_load
from .generic import LogitCalibrator
from .universal_calibration_models import TieredTSModel


def std_proc(std, p=4):
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
    }
    df_bot = {
        "token_ids": [],
        "token_values": [],
    }
    total_tokens = sum([len(v) for _, v in token_confs.items()])

    for k, v in token_confs.items():
        v = torch.Tensor(v)
        n_response_occurrences = len(response_occurrences[k])
        df_top["token_ids"].append(k)
        df_bot["token_ids"].append(k)

        m = torch.mean(v)
        s = torch.std(v, correction=0 if len(v) == 1 else 1)
        fv = len(v) / total_tokens
        r = n_response_occurrences / len(calibration_dset)

        df_top["token_values"].append(metric_func(m, s, fv, r).item())
        df_bot["token_values"].append(metric_func((1 - m), s, fv, r).item())

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
    def __init__(self, llm_bundle, top_k=10, bot_k=10, _calibrator_model=None):
        if _calibrator_model is None:
            _calibrator_model = TieredTSModel()

        self.top_k = top_k
        self.bot_k = bot_k
        self.top_token_values = self.top_token_ids = self.bot_token_values = self.bot_token_ids = None

        super().__init__(llm_bundle, _calibrator_model)

    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        _, _ = self.compute_scores_and_indices(calibration_dset)

        self.calibrator_model.set_tokens(self.top_token_ids[:self.top_k], self.bot_token_ids[:self.bot_k])
        super().calibrate(calibration_dset, **kwargs)

    def compute_scores_and_indices(self, calibration_dset: DictDataset):
        df_top, df_bot = compute_top_bot_dfs(calibration_dset, self.llm_bundle, self.__compute_metric)

        self.top_token_ids = torch.as_tensor(df_top["token_ids"].to_numpy())
        self.top_token_values = torch.as_tensor(df_top["token_values"].to_numpy())
        self.bot_token_ids = torch.as_tensor(df_bot["token_ids"].to_numpy())
        self.bot_token_values = torch.as_tensor(df_bot["token_values"].to_numpy())

        return df_top, df_bot

    def __compute_metric(self, mean, std, relative_token_frequency, response_frequency_ratio):
        #f = lambda x: -1 / (x / 4 + 1) + 1
        sf_std = std_proc(std)
        #f_tf = f(token_frequency)
        return mean * sf_std * relative_token_frequency * (response_frequency_ratio ** 10)

    def load(self, filepath):
        d = dill_load(filepath)
        self.top_k = d["top_k"]
        self.bot_k = d["bot_k"]
        self.top_token_ids = d["top_token_ids"]
        self.top_token_values = d["top_token_values"]
        self.bot_token_ids = d["bot_token_ids"]
        self.bot_token_values = d["bot_token_values"]

        self.calibrator_model.set_tokens(self.top_token_ids[:self.top_k], self.bot_token_ids[:self.bot_k])
        super().load(filepath)

    def save(self, filepath, **kwargs):
        _other_entries = self.get_frequency_dict()
        super().save(filepath, _other_entries)

    def get_frequency_dict(self):
        assert self.top_k is not None
        assert self.bot_k is not None
        assert self.top_token_ids is not None
        assert self.bot_token_ids is not None
        assert self.top_token_values is not None
        assert self.bot_token_values is not None

        return {
            "top_k": self.top_k,
            "bot_k": self.bot_k,
            "top_token_ids": self.top_token_ids,
            "top_token_values": self.top_token_values,
            "bot_token_ids": self.bot_token_ids,
            "bot_token_values": self.bot_token_values
        }


class FrequencyTSTopOnly(FrequencyTS):
    def __init__(self, llm_bundle, top_k=10):
        super().__init__(llm_bundle, top_k, bot_k=0)


class FrequencyTSBotOnly(FrequencyTS):
    def __init__(self, llm_bundle, bot_k=10):
        super().__init__(llm_bundle, top_k=0, bot_k=bot_k)


class FrequencyTSMeanOnly(FrequencyTS):
    """
    FrequencyTSModel that only considers the mean token confidence. Does not factor in anything else.
    """
    def __compute_metric(self, mean, std, relative_token_frequency, response_frequency_ratio):
        return mean


class FrequencyTSMeanStdOnly(FrequencyTS):
    """
    FrequencyTSModel that only considers the mean token confidence and their stds. Does not factor in anything else.
    """
    def __compute_metric(self, mean, std, token_frequency, response_frequency_ratio):
        return mean * std_proc(std)


class FrequencyTSNoRF(FrequencyTS):
    """
    FrequencyTSModel without response frequency ratio.
    """
    def __compute_metric(self, mean, std, relative_token_frequency, response_frequency_ratio):
        return mean * std_proc(std) * relative_token_frequency


class FrequencyTSNoTF(FrequencyTS):
    """
    FrequencyTSModel without token frequency.
    """
    def __compute_metric(self, mean, std, token_frequency, response_frequency_ratio):
        return mean * std_proc(std) * (response_frequency_ratio ** 10)
