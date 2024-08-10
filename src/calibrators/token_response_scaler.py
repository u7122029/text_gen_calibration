import warnings

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DictDataset
from utils import DEVICE, dill_load, dill_save
from .generic import Calibrator
from .universal_calibration_models import TokenCalibratorModel


class TokenCalibrator(Calibrator):
    def __init__(self, llm_bundle, label_key="correct"):
        super().__init__(llm_bundle)
        self.calibrator_model = TokenCalibratorModel(device=DEVICE)
        self.label_key = label_key
        self.loss_fn = nn.MSELoss()

    def calibration_epoch(self, pbar, postfix, optimiser, **kwargs):
        postfix["total_loss_last_epoch"] = 0
        for batch in pbar:
            label_batch = torch.Tensor(batch[self.label_key]).to(DEVICE)
            tokens_batch = batch["tokens"]
            question_batch = batch["question"]

            response_batch = self.llm_bundle.tokeniser.batch_decode(tokens_batch, skip_special_tokens=True)
            assert len(question_batch) == len(response_batch)
            inputs_batch = [f"{question}\n{response}" for question, response in zip(question_batch, response_batch)]

            optimiser.zero_grad()
            out_token_confs = self.calibrator_model(inputs_batch)
            loss = self.loss_fn(out_token_confs, label_batch)
            loss.backward()
            optimiser.step()
            postfix["total_loss_last_epoch"] += loss.item()

    def calibrate(self, calibration_dset: DictDataset, batch_size=1, epochs=30, lr=0.01, **kwargs) -> None:
        calibration_dl = DataLoader(calibration_dset,
                                    collate_fn=calibration_dset.collate_fn("question", "tokens", self.label_key),
                                    batch_size=batch_size,
                                    shuffle=True)
        # Optimise llm.
        self.calibrator_model = self.calibrator_model.to(DEVICE)
        self.loss_fn.to(DEVICE)

        optimiser = optim.SGD(self.calibrator_model.parameters(), lr=lr)

        print("Training Calibrator")
        self.calibrator_model.train()

        postfix = {}
        for epoch_idx in range(epochs):
            pbar = tqdm(calibration_dl,
                        desc=f"Epoch {epoch_idx + 1}/{epochs}",
                        postfix=postfix)

            self.calibration_epoch(pbar, postfix, optimiser)

        calibration_dset["calibrated_successful"] = torch.ones(len(calibration_dset)).bool()
        self.calibrator_model.eval()
        self.tuned = True

    def load(self, filepath):
        self.calibrator_model.load_state_dict(dill_load(filepath)["state_dict"])
        self.calibrator_model.eval()
        self.tuned = True

    def save(self, filepath, _other_entries=None):
        if _other_entries is None:
            _other_entries = {}
        _other_entries.update({"state_dict": self.calibrator_model.state_dict()})
        dill_save(_other_entries, filepath)

    def test_loop(self, test_dset):
        confs_after_calibration = []
        for batch in tqdm(test_dset):
            input_batch = [f"{batch["question"]}\n{self.llm_bundle.tokeniser.decode(batch["tokens"], skip_special_tokens=True)}"]
            confs = self.calibrator_model(input_batch).cpu()
            confs_after_calibration.extend(confs)
        return confs_after_calibration

    def test(self, test_dset: DictDataset, **kwargs):
        if not self.tuned:
            warnings.warn("Calibrator model has not been loaded or trained. Expect dubious results.")

        self.calibrator_model = self.calibrator_model.to(DEVICE)
        with torch.no_grad():
            confs_after_calibration = self.test_loop(test_dset)

        out_dict = {
            "calibrated_confs": torch.Tensor(confs_after_calibration),
            "calibrated_successful": torch.ones(len(test_dset)).bool()
        }
        return out_dict