from typing import List

import torch
from torch import nn
from data import DictDataset
from .generic import Calibrator, LogitTokenToConfidenceCalibrator
from utils import TextGenLLMBundle, dill_load


class FrequencyTS(LogitTokenToConfidenceCalibrator):
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

        def get_save_dict(self):
            return {
                "top_token_ids": self.top_token_ids,
                "bot_token_ids": self.bot_token_ids
            }

    def __init__(self, llm_bundle):
        self.top_k_token_ids = None
        self.bot_k_token_ids = None
        super().__init__(llm_bundle, FrequencyTS.TSModel())

    def calibrate(self, calibration_dset: DictDataset, top_k=1, bot_k=1, **kwargs):
        # we first need to generate the frequency scores.
        vocab_size = self.llm_bundle.vocab_size()
        token_frequencies = torch.zeros(vocab_size) # how many times does the token appear in every response?
        token_response_frequencies = torch.zeros(vocab_size) # how many responses does the token appear at least once in?
        token_response_frequencies2 = torch.zeros(vocab_size) # same as above, but takes squared values
        token_total_confs = torch.zeros(vocab_size)

        for item in calibration_dset:
            tokens = item["tokens"]
            token_counts = torch.bincount(tokens, minlength=vocab_size)
            token_frequencies += token_counts
            token_occurrences = (token_counts > 0).int()
            token_response_frequencies += token_occurrences
            token_response_frequencies2 += token_occurrences ** 2

            prob_vecs = torch.softmax(item["logits"], dim=1)
            token_confs = torch.take_along_dim(prob_vecs, tokens.unsqueeze(1), dim=1).squeeze(1)
            token_total_confs = token_total_confs.scatter_add(0, tokens, token_confs)

        token_total_confs_nonzero = token_total_confs.clone()
        token_total_confs_nonzero[token_total_confs_nonzero > 0] += 1

        mean_token_frequencies = token_response_frequencies / token_total_confs_nonzero
        std_token_frequencies = torch.sqrt(
            (token_response_frequencies2 - 2*token_response_frequencies*mean_token_frequencies + mean_token_frequencies ** 2)
            / torch.relu(token_total_confs_nonzero - 1)
        )

        s = lambda x: -2*x + 1
        f = lambda x: -1/(x/4 + 1) + 1
        high_token_scores = (mean_token_frequencies
                             * s(std_token_frequencies)
                             * f(token_frequencies)
                             * token_response_frequencies / len(calibration_dset))
        top_token_values, top_token_ids = torch.sort(high_token_scores)

        bot_token_scores = ((1 - mean_token_frequencies)
                            * s(std_token_frequencies)
                            * f(token_frequencies)
                            * token_response_frequencies / len(calibration_dset))
        bot_token_values, bot_token_ids = torch.sort(bot_token_scores)

        self.top_k_token_ids = top_token_ids[-top_k:]
        self.bot_k_token_ids = bot_token_ids[-bot_k:]

        self.calibrator_model.set_tokens(self.top_k_token_ids, self.bot_k_token_ids)
        super().calibrate(calibration_dset, **kwargs)

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
        self.calibrator_model.set_tokens(d["top_token_ids"], d["bot_token_ids"])
        super().load(filepath)

    def save(self, filepath, **kwargs):
        super().save(filepath, self.calibrator_model.get_save_dict())