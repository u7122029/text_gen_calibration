import torch
from torch import nn
from tqdm import tqdm
from abc import ABC, abstractmethod

from collate_postprocess_functions import lhs_token_repeat_label_key
from data import DictDataset
from llm_models.generic import LLMBundle
from utils import DEVICE
from .generic import LogitCalibrator
from utils.loss_functions import CorrectnessAwareLoss, CrossUnifEntropy


class LHSModel(nn.Module):
    def __init__(self, llm_bundle: LLMBundle):
        super().__init__()
        self.llm_bundle = llm_bundle
        self.llm_bundle.load_model(silent=True)
        self.fc = nn.Linear(in_features=llm_bundle.llm_model.config.hidden_size, out_features=1)

    def forward(self, x, tokens=None):
        # x.shape: [hidden_feature_vecs, num_hidden_features]
        # tokens.shape: [hidden_feature_vecs]

        temperatures = nn.functional.softplus(self.fc(x)) # vector of temperatures.
        temperatures = torch.clip(temperatures, min=1e-5)

        # lm head will not be trained, so .float() and device changes are safe.
        logits = self.llm_bundle.final_hs_to_logits(x).float().to(temperatures.device)
        logits = logits / temperatures

        #prob_vecs = torch.softmax(logits, dim=1)
        #if tokens is not None:
        #    confs = torch.take_along_dim(prob_vecs, tokens.unsqueeze(1), dim=1).squeeze(1)
        #else:
        #    confs = torch.max(x, dim=1).values
        return logits # [confs]


class LastHiddenStateCalibrator(LogitCalibrator, ABC):
    @abstractmethod
    def __init__(self, llm_bundle, loss_fn):
        super().__init__(llm_bundle, LHSModel(llm_bundle), loss_fn, "final_hidden_states")

    def calibrate(self, calibration_dset: DictDataset, *args, **kwargs):
        super().calibrate(calibration_dset, *args, _postprocess_fn=lhs_token_repeat_label_key(self.label_key))

    def test_loop(self, test_dset):
        print("here")
        confs_after_calibration = []
        for batch in tqdm(test_dset):
            final_hidden_states = batch[self.input_key].to(DEVICE).float()
            #tokens = batch["tokens"].to(DEVICE)
            token_confs = self.calibrator_model(final_hidden_states).cpu()
            out = torch.mean(token_confs)
            confs_after_calibration.append(out)
        return confs_after_calibration


class LHS_CA(LastHiddenStateCalibrator):
    """
    Last Hidden State Calibrator with Correctness-Aware Loss
    """
    def __init__(self, llm_bundle):
        super().__init__(llm_bundle, CorrectnessAwareLoss(nn.MSELoss()))


class LHS_BCE(LastHiddenStateCalibrator):
    """
    Last Hidden State Calibrator with Correctness-Aware Binary Cross Entropy Loss
    """
    def __init__(self, llm_bundle):
        super().__init__(llm_bundle, CorrectnessAwareLoss(nn.BCELoss()))


class LHS_Paper(LastHiddenStateCalibrator):
    """
    Last Hidden State Calibrator with loss function based on Xie et. al.
    CALIBRATING LANGUAGE MODELS WITH ADAPTIVE TEMPERATURE SCALING
    """
    def __init__(self, llm_bundle):
        super().__init__(llm_bundle, CrossUnifEntropy())

    def calibration_epoch(self, pbar, postfix, optimiser, **kwargs):
        postfix["total_loss_last_epoch"] = 0
        torch.autograd.set_detect_anomaly(True)

        for i, batch in enumerate(pbar):
            #label_batch = batch[self.label_key].to(DEVICE)
            logits_batch = batch[self.input_key].to(DEVICE).float()
            tokens_batch = batch["tokens"].to(DEVICE)

            optimiser.zero_grad()

            out_token_confs = self.calibrator_model(logits_batch)
            #label_batch = label_batch.to(out_token_confs.dtype)
            loss = self.loss_fn(out_token_confs, tokens_batch)
            print(loss)
            loss.backward()
            optimiser.step()
            postfix["total_loss_last_epoch"] += loss.item()
