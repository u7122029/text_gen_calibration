from typing import List

import pandas as pd
import torch
from torch import nn

from data import DictDataset
from utils import dill_load
from .generic import LogitTokenToConfidenceCalibrator


class FrequencyTS(LogitTokenToConfidenceCalibrator):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    class TSModel(nn.Module):
        """
        Contains 3 temperature parameters.
        One determines the adjustment of the token ids that commonly occur with high confidence
        One determines the adjustment of the token ids that commonly occur with low confidence
        The last is a general temperature that adjusts all the tokens after adjustment from the previous two temps.
        """
        def __init__(self):
            super().__init__()
            self.top_token_ids = None
            self.bot_token_ids = None

            self.top_temp = nn.Parameter(torch.tensor(1.0))
            self.bot_temp = nn.Parameter(torch.tensor(1.0))
            self.general_temp = nn.Parameter(torch.tensor(1.0))

        def forward(self, x, tokens=None):
            assert self.top_token_ids is not None
            assert self.bot_token_ids is not None

            # x.shape: [logit_vec, vocab size]
            x[:,self.top_token_ids] = x[:,self.top_token_ids] / self.top_temp
            x[:,self.bot_token_ids] = x[:,self.bot_token_ids] / self.bot_temp
            x = x / self.general_temp
            x = torch.softmax(x, dim=1)
            if tokens is not None:
                x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
            else:
                x = torch.max(x, dim=1).values
            return x  # [confs]

        def set_tokens(self, top_token_ids: torch.Tensor, bot_token_ids: torch.Tensor):
            self.top_token_ids = top_token_ids
            self.bot_token_ids = bot_token_ids

    def __init__(self, llm_bundle):
        self.top_k = None
        self.bot_k = None
        self.top_token_values = self.top_token_ids = self.bot_token_values = self.bot_token_ids = None

        super().__init__(llm_bundle, FrequencyTS.TSModel())

    def calibrate(self, calibration_dset: DictDataset, top_k=1, bot_k=1, **kwargs):
        self.compute_scores_and_indices(calibration_dset, top_k, bot_k)

        self.calibrator_model.set_tokens(self.top_token_ids[-self.top_k:], self.bot_token_ids[-self.bot_k:])
        super().calibrate(calibration_dset, **kwargs)

    def compute_scores_and_indices(self, calibration_dset: DictDataset, top_k=1, bot_k=1):
        self.top_k = top_k
        self.bot_k = bot_k

        # we first need to generate the frequency scores.
        vocab_size = self.llm_bundle.vocab_size()
        token_frequencies = torch.zeros(vocab_size)  # how many times does the token appear in every response?
        token_response_frequencies = torch.zeros(
            vocab_size)  # how many responses does the token appear at least once in?
        token_response_frequencies2 = torch.zeros(vocab_size)  # same as above, but takes squared values
        token_total_confs = torch.zeros(vocab_size)

        for item in calibration_dset:
            tokens = item["tokens"].long()
            token_counts = torch.bincount(tokens, minlength=vocab_size)
            token_frequencies += token_counts
            token_occurrences = (token_counts > 0).int()
            token_response_frequencies += token_occurrences
            token_response_frequencies2 += token_occurrences ** 2

            prob_vecs = torch.softmax(item["logits"], dim=1)
            token_confs = torch.take_along_dim(prob_vecs, tokens.unsqueeze(1), dim=1).squeeze(1)
            token_total_confs = token_total_confs.scatter_add(0, tokens, token_confs)

        token_frequencies_nonzero = token_frequencies.clone()
        token_frequencies_nonzero[token_frequencies_nonzero == 0] += 1

        mean_token_frequencies = token_total_confs / token_frequencies_nonzero
        std_token_frequencies = torch.sqrt(
            (token_response_frequencies2 - 2 * token_response_frequencies * mean_token_frequencies + mean_token_frequencies ** 2)
            / token_frequencies_nonzero
        )
        print(mean_token_frequencies)
        print(std_token_frequencies)
        print(token_frequencies)
        print(token_response_frequencies)
        s = lambda x: -2 * x + 1
        f = lambda x: -1 / (x / 4 + 1) + 1
        high_token_scores = (mean_token_frequencies
                             * s(std_token_frequencies)
                             * f(token_frequencies)
                             * token_response_frequencies / len(calibration_dset))
        self.top_token_values, self.top_token_ids = torch.sort(high_token_scores)

        bot_token_scores = ((1 - mean_token_frequencies)
                            * s(std_token_frequencies)
                            * f(token_frequencies)
                            * token_response_frequencies / len(calibration_dset))
        self.bot_token_values, self.bot_token_ids = torch.sort(bot_token_scores)

    def collate_fn(self, data_list):
        out = {
            "logits": [],
            "tokens": [],
            "correct": []
        }
        for d in data_list:
            out["logits"].append(d["logits"])
            out["tokens"].append(d["tokens"])
            out["correct"].append(d["correct"].repeat(len(d["tokens"])))

        out["logits"] = torch.cat(out["logits"], dim=0)
        out["tokens"] = torch.cat(out["tokens"])
        out["correct"] = torch.cat(out["correct"]).float()
        return out

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

        self.calibrator_model.set_tokens(self.top_token_ids[-self.top_k:], self.bot_token_ids[-self.bot_k:])
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

    def compile_token_score(self):
        df = self.get_frequency_dict()
        df["top_token_ids"] = self.llm_bundle.tokeniser.batch_decode(df["top_token_ids"])
        df["bot_token_ids"] = self.llm_bundle.tokeniser.batch_decode(df["bot_token_ids"])
        del df["top_k"]
        del df["bot_k"]
        df = pd.DataFrame(df)
        return df