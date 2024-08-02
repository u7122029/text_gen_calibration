from abc import ABC, abstractmethod

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


class LogitTokenToConfidenceCalibrator(Calibrator, ABC):
    @abstractmethod
    def __init__(self, llm_bundle, calibrator_model, label_key="correct", loss_fn=None):
        super().__init__(llm_bundle)
        if loss_fn is None:
            self.loss_fn = nn.MSELoss() # calibration aware loss with l2 norm squared.
        else:
            self.loss_fn = loss_fn

        self.calibrator_model = calibrator_model.to(DEVICE)
        self.label_key = label_key
        self.tuned = False

    def calibration_epoch(self, pbar, postfix, optimiser, **kwargs):
        postfix["total_loss_last_epoch"] = 0
        for batch in pbar:
            label_batch = batch[self.label_key].to(DEVICE)
            logits_batch = batch["logits"].to(DEVICE)
            tokens_batch = batch["tokens"].to(DEVICE)

            optimiser.zero_grad()
            out_token_confs = self.calibrator_model(logits_batch, tokens_batch)
            loss = self.loss_fn(out_token_confs, label_batch)
            loss.backward()
            optimiser.step()
            postfix["total_loss_last_epoch"] += loss.item()

    def __collate_post_process(self, out_dict: dict):
        out_dict["logits"] = torch.cat(out_dict["logits"], dim=0)
        out_dict[self.label_key] = torch.cat(
            [c.repeat(len(t)) for c, t in zip(out_dict[self.label_key], out_dict["tokens"])]).float()
        out_dict["tokens"] = torch.cat(out_dict["tokens"])
        return out_dict

    def calibrate(self,
                  calibration_dset: DictDataset,
                  batch_size=1,
                  epochs=30,
                  lr=0.01,
                  _postprocess_fn=None,
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
        if _postprocess_fn is None:
            _postprocess_fn = self.__collate_post_process

        calibration_dl = DataLoader(calibration_dset,
                                    collate_fn=calibration_dset.collate_fn("logits", "tokens", self.label_key,
                                                                           postprocess_fn=_postprocess_fn),
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

        calibration_dset.add_column("calibrated_successful", torch.ones(len(calibration_dset)).bool())
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

        out_dict = {
            "calibrated_confs": torch.Tensor(confs_after_calibration),
            "calibrated_successful": torch.ones(len(test_dset)).bool()
        }
        return out_dict