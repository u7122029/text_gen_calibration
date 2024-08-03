from typing import List

import pandas as pd
import torch

from data import DictDataset
from utils import dill_load
from .generic import LogitCalibrator
from .universal_calibration_models import TieredTSModel


class FrequencyTS(LogitCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle):
        self.top_k = None
        self.bot_k = None
        self.top_token_values = self.top_token_ids = self.bot_token_values = self.bot_token_ids = None

        super().__init__(llm_bundle, TieredTSModel())

    def calibrate(self, calibration_dset: DictDataset, top_k=10, bot_k=10, **kwargs):
        _, _ = self.compute_scores_and_indices(calibration_dset, top_k, bot_k)

        self.calibrator_model.set_tokens(self.top_token_ids[:self.top_k], self.bot_token_ids[:self.bot_k])
        super().calibrate(calibration_dset, **kwargs)

    def compute_scores_and_indices(self, calibration_dset: DictDataset, top_k=1, bot_k=1):
        self.top_k = top_k
        self.bot_k = bot_k

        token_confs = {}
        response_occurrences = {}

        for i, item in enumerate(calibration_dset):
            tokens = item["tokens"].long()
            prob_vecs = torch.softmax(item["logits"], dim=1)
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

        sf = lambda x: -2 * x + 1
        f = lambda x: -1 / (x / 4 + 1) + 1

        df_top = {
            "token_ids": [],
            "token_values": [],
        }
        df_bot = {
            "token_ids": [],
            "token_values": [],
        }
        for k, v in token_confs.items():
            v = torch.Tensor(v)
            n_response_occurrences = len(response_occurrences[k])
            df_top["token_ids"].append(k)
            df_bot["token_ids"].append(k)

            m = torch.mean(v)
            s = sf(torch.std(v, correction=0 if len(v) == 1 else 1))
            fv = f(len(v))
            r = n_response_occurrences / len(calibration_dset)

            df_top["token_values"].append((m * s * fv * r).item())
            df_bot["token_values"].append(((1 - m) * s * fv * r).item())

        df_top["token_str"] = self.llm_bundle.tokeniser.batch_decode(df_top["token_ids"])
        df_bot["token_str"] = self.llm_bundle.tokeniser.batch_decode(df_bot["token_ids"])

        df_top = pd.DataFrame(df_top).sort_values("token_values", ascending=False)
        df_bot = pd.DataFrame(df_bot).sort_values("token_values", ascending=False)

        self.top_token_ids = torch.as_tensor(df_top["token_ids"].to_numpy())
        self.top_token_values = torch.as_tensor(df_top["token_values"].to_numpy())
        self.bot_token_ids = torch.as_tensor(df_bot["token_ids"].to_numpy())
        self.bot_token_values = torch.as_tensor(df_bot["token_values"].to_numpy())

        return df_top, df_bot

    def compute_scores_and_indices1(self, calibration_dset: DictDataset, top_k=1, bot_k=1):
        self.top_k = top_k
        self.bot_k = bot_k

        # we first need to generate the frequency scores.
        vocab_size = self.llm_bundle.vocab_size()
        token_frequencies = torch.zeros(vocab_size) # how many times does the token appear in every response?
        token_response_frequencies = torch.zeros(vocab_size) # how many responses does the token appear at least once in?
        token_total_confs = torch.zeros(vocab_size)
        token_total_confs2 = torch.zeros(vocab_size)

        for item in calibration_dset:
            tokens = item["tokens"].long()
            token_counts = torch.bincount(tokens, minlength=vocab_size)
            token_frequencies += token_counts
            token_occurrences = (token_counts > 0).int()
            token_response_frequencies += token_occurrences
            #token_response_frequencies2 += token_occurrences ** 2

            prob_vecs = torch.softmax(item["logits"], dim=1)
            token_confs = torch.take_along_dim(prob_vecs, tokens.unsqueeze(1), dim=1).squeeze(1)
            token_total_confs = token_total_confs.scatter_add(0, tokens, token_confs)
            token_total_confs2 = token_total_confs2.scatter_add(0, tokens, token_confs**2)

        token_frequencies_nonzero = token_frequencies.clone()
        token_frequencies_nonzero[token_frequencies_nonzero == 0] = 1

        mean_token_confs = token_total_confs / token_frequencies_nonzero
        std_token_frequencies = torch.sqrt(
            (token_total_confs2 - 2 * token_total_confs * mean_token_confs + mean_token_confs ** 2)
            / token_frequencies_nonzero
        )

        s = lambda x: -2 * x + 1
        f = lambda x: -1 / (x / 4 + 1) + 1
        high_token_scores = (mean_token_confs
                             * s(std_token_frequencies)
                             * f(token_frequencies)
                             * token_response_frequencies / len(calibration_dset))
        self.top_token_values, self.top_token_ids = torch.sort(high_token_scores, descending=True)

        bot_token_scores = ((1 - mean_token_confs)
                            * s(std_token_frequencies)
                            * f(token_frequencies)
                            * token_response_frequencies / len(calibration_dset))
        self.bot_token_values, self.bot_token_ids = torch.sort(bot_token_scores, descending=True)

    def dset_columns(self) -> List[str]:
        return ["tokens", "logits", "correct"]

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