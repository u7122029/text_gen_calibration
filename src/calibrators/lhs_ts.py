import torch
from torch import nn
from tqdm import tqdm

from collate_postprocess_functions import lhs_token_repeat_label_key
from data import DictDataset
from llm_models.generic import LLMBundle
from utils import DEVICE
from .generic import LogitCalibrator


class LHSModel(nn.Module):
    def __init__(self, llm_bundle: LLMBundle):
        super().__init__()
        self.llm_bundle = llm_bundle
        if llm_bundle.hidden_features is None:
            #print("HERE!!!!")
            llm_bundle.load_model(silent=False, lm_head_only=True)
        #print("RIGHT BEFORE SELF.FC")
        self.fc = nn.Linear(in_features=llm_bundle.hidden_features, out_features=1)

    def temp_scale(self, x):
        temperatures = nn.functional.softplus(self.fc(x))  # vector of temperatures.
        torch.clip_(temperatures, min=1e-5)

        # lm head will not be trained, so .float() and device changes are safe.
        logits = self.llm_bundle.final_hs_to_logits(x).float().to(temperatures.device)
        logits.div_(temperatures)

        return logits

    def forward(self, x, tokens=None):
        # x.shape: [hidden_feature_vecs, num_hidden_features]
        # tokens.shape: [hidden_feature_vecs]

        x = self.temp_scale(x)

        x = torch.softmax(x, dim=1)
        if tokens is not None:
            confs = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
        else:
            confs = torch.max(x, dim=1).values
        return confs # [confs]


class LastHiddenStateCalibrator(LogitCalibrator):
    def __init__(self, llm_bundle, loss_fn, _calibrator_model=None):
        if _calibrator_model is None:
            _calibrator_model = LHSModel(llm_bundle)
        super().__init__(llm_bundle, _calibrator_model, loss_fn, "final_hidden_states")

    def calibrate(self, calibration_dset: DictDataset, *args, **kwargs):
        super().calibrate(calibration_dset, *args, _postprocess_fn=lhs_token_repeat_label_key(self.label_key), **kwargs)

    def test_loop(self, test_dset):
        response_confs_after_calib = []
        token_confs_after_calib = []
        for batch in tqdm(test_dset):
            final_hidden_states = batch[self.input_key].to(DEVICE).float()
            tokens = batch["tokens"].to(DEVICE)
            token_confs = self.calibrator_model(final_hidden_states, tokens).cpu()
            token_confs_after_calib.append(token_confs)
            response_confs_after_calib.append(torch.mean(token_confs))
        return response_confs_after_calib, token_confs_after_calib
