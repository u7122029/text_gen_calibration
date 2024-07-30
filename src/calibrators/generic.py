from abc import ABC, abstractmethod
from typing import Tuple, Iterable, List

from data import DictDataset
from utils import DEVICE, LLMBundle, dill_save, dill_load
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import torch, warnings


class Calibrator(ABC):
    """
    To use a Calibrator, you must

    1. Run the calibrate method to generate a new instance of the given calibrator, or load the calibrator llm weights to generate the calibrator llm.

    2. Run the test method to test the calibrator on some data.

    You may use the save method to save the parts of the calibrator that require persistence.
    Note that the save method may not do anything at all if there is nothing to save.
    """
    def __init__(self, llm_bundle: LLMBundle):
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
    def save(self, filepath, **kwargs):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


class LogitTokenToConfidenceCalibrator(Calibrator):
    def __init__(self, llm_bundle, calibrator_model):
        super().__init__(llm_bundle)
        self.calibrator_model = calibrator_model.to(DEVICE)
        self.tuned = False

    def calibration_epoch(self, pbar, postfix, optimiser, loss_fn, **kwargs):
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

    @staticmethod
    def __collate_post_process(out_dict):
        out_dict["logits"] = torch.cat(out_dict["logits"], dim=0)
        out_dict["tokens"] = torch.cat(out_dict["tokens"])
        out_dict["correct"] = torch.cat(out_dict["correct"]).float()
        return out_dict

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
        calibration_dl = DataLoader(calibration_dset,
                                    collate_fn=calibration_dset.collate_fn(*self.dset_columns(),
                                                                           postprocess_fn=self.__collate_post_process),
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

            self.calibration_epoch(pbar, postfix, optimiser, loss_fn)
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
            logits = batch["logits"].to(DEVICE)
            tokens = batch["tokens"].to(DEVICE)
            token_confs = self.calibrator_model(logits, tokens).cpu()
            out = torch.mean(token_confs)
            confs_after_calibration.append(out)
        return confs_after_calibration

    def test(self, test_dset: DictDataset, **kwargs):
        if not self.tuned:
            warnings.warn("Calibrator model has not been loaded or trained. Expect dubious results.")

        self.calibrator_model = self.calibrator_model.to(DEVICE)
        with torch.no_grad():
            confs_after_calibration = self.test_loop(test_dset)
        return torch.Tensor(confs_after_calibration)