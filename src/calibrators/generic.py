from abc import ABC, abstractmethod
from typing import Tuple, Iterable, List

from data import DictDataset
from utils import DEVICE, TokenLogitsDataset
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import torch, warnings
from datasets import Dataset


class Calibrator(ABC):
    """
    To use a Calibrator, you must

    1. Run the calibrate method to generate a new instance of the given calibrator, or load the calibrator llm weights to generate the calibrator llm.

    2. Run the test method to test the calibrator on some data.

    You may use the save method to save the parts of the calibrator that require persistence.
    Note that the save method may not do anything at all if there is nothing to save.
    """
    def __init__(self, llm_bundle):
        self.llm_bundle = llm_bundle
        self.calibrator_model = None

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def calibrate(self, calibration_dset: DictDataset, **kwargs):
        pass

    @abstractmethod
    def test(self, test_dset: DictDataset, **kwargs):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass

    @abstractmethod
    def dset_columns(self) -> List[str]:
        """
        Selects the columns to be used for calibration.
        :return:
        """
        pass

    @abstractmethod
    def collate_fn(self, data_list):
        pass


class LogitTokenToConfidenceCalibrator(Calibrator):
    def __init__(self, llm_bundle, calibrator_model):
        super().__init__(llm_bundle)
        self.calibrator_model = calibrator_model.to(DEVICE)
        self.tuned = False

    def get_dataset(self, tokens, logits, correct, **kwargs):
        return TokenLogitsDataset(logits, tokens, correct)

    def calibration_step(self, pbar, postfix, optimiser, loss_fn, **kwargs):
        postfix["total_loss_last_epoch"] = 0
        for batch in pbar:
            is_correct_batch = batch["correct"].to(DEVICE)
            logits_batch = batch["logits"].to(DEVICE)

            optimiser.zero_grad()
            out_token_confs = self.calibrator_model(logits_batch)
            loss = loss_fn(out_token_confs, is_correct_batch)
            loss.backward()
            optimiser.step()
            postfix["total_loss_last_epoch"] += loss.item()

    def dset_columns(self) -> List[str]:
        return ["logits", "tokens", "correct"]

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

    def calibrate(self,
                  calibration_dset: DictDataset,
                  batch_size=1,
                  epochs=30,
                  lr=0.01,
                  **kwargs):
        """
        Calibrates the calibrator model. By default, this will use the TokenLogitsDataset. You will need to override
        this function if you want to use a different dataset.
        :param batch_size:
        :param calibration_dset:
        :param epochs:
        :param lr:
        :param kwargs:
        :return:
        """
        # Assume calib_logits has shape [dset_length, response_length, vocab_size]
        calibration_dset.restrict_keys(self.dset_columns())
        calibration_dl = DataLoader(calibration_dset,
                                    collate_fn=self.collate_fn,
                                    batch_size=batch_size,
                                    shuffle=True)
        print("Made dataloader.")
        # Optimise llm.
        self.calibrator_model = self.calibrator_model.to(DEVICE)
        loss_fn = nn.MSELoss().to(DEVICE) # calibration aware loss with l2 norm squared.
        optimiser = optim.SGD(self.calibrator_model.parameters(), lr=lr)

        print("Training Calibrator")
        self.calibrator_model.train()

        postfix = {}
        for epoch_idx in range(epochs):
            pbar = tqdm(calibration_dl,
                        desc=f"Epoch {epoch_idx + 1}/{epochs}",
                        postfix=postfix)

            self.calibration_step(pbar, postfix, optimiser, loss_fn)
        self.calibrator_model.eval()
        self.tuned = True
        calibration_dset.reset_keys()

    def load(self, filepath):
        self.calibrator_model.load_state_dict(torch.load(filepath))
        self.calibrator_model.eval()

    def save(self, filepath):
        torch.save(self.calibrator_model.state_dict(), filepath)

    def test_loop(self, test_dset):
        confs_after_calibration = []
        for batch in tqdm(test_dset):
            logits = batch["logits"].to(DEVICE)
            tokens = batch["tokens"].to(DEVICE)
            token_confs = self.calibrator_model(logits, tokens).cpu()
            out = torch.mean(token_confs)
            confs_after_calibration.append(out)
        return confs_after_calibration

    def test(self, test_dset: DictDataset, **kwargs):
        test_dset.restrict_keys(self.dset_columns())
        if not self.tuned:
            warnings.warn("Calibrator model has not been loaded or trained. Expect dubious results.")

        self.calibrator_model = self.calibrator_model.to(DEVICE)
        with torch.no_grad():
            confs_after_calibration = self.test_loop(test_dset)
        test_dset.reset_keys()
        return torch.Tensor(confs_after_calibration)