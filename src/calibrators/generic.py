from abc import ABC, abstractmethod
from typing import Optional

from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import torch, warnings

from utils import EarlyStopping, LossFunc, LossFunctionDetails
from collate_postprocess_functions import logit_token_repeat_label_key
from data import DictDataset
from utils import DEVICE, dill_save, dill_load
from llm_models.generic import LLMBundle


class Calibrator(ABC):
    """
    To use a Calibrator, you must

    1. Run the calibrate method to generate a new instance of the given calibrator, or load the calibrator llm weights to generate the calibrator llm.

    2. Run the test method to test the calibrator on some data.

    You may use the save method to save the parts of the calibrator that require persistence.
    Note that the save method may not do anything at all if there is nothing to save.
    """

    def __init__(self,
                 llm_bundle: LLMBundle,
                 loss_fn: LossFunctionDetails,
                 calibrator_model: Optional[nn.Module] = None,
                 learning_rate: Optional[float] = None):
        if isinstance(calibrator_model, nn.Module):
            calibrator_model.eval()

        self.__llm_bundle = llm_bundle
        self.__loss_fn = loss_fn.loss_fn

        self.__learning_rate = learning_rate
        if calibrator_model is not None and learning_rate is None:
            # Learning rate for largest model (Token Calibrator)
            #const = torch.log(torch.tensor(5)) / torch.log(torch.tensor(184423682))

            # Interpolate learning rate for all other models.
            """self.__learning_rate = torch.round(
                1e-2 *
                (sum(p.numel() for p in calibrator_model.parameters() if p.requires_grad)) ** (-const),
                decimals=7)"""
            self.__learning_rate = 0.01
        print(f"Learning rate: {self.__learning_rate}")
        self.__calibrator_model = calibrator_model

        self.loss_fn.to(DEVICE)
        self.tuned = False

    @property
    def llm_bundle(self):
        return self.__llm_bundle

    @property
    def loss_fn(self):
        return self.__loss_fn

    @property
    def learning_rate(self):
        return self.__learning_rate

    @property
    def calibrator_model(self):
        return self.__calibrator_model

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def calibrate(self, calibration_dset: DictDataset, **kwargs) -> None:
        pass

    @abstractmethod
    def test_loop(self, test_dset: DictDataset) -> tuple:
        """
        Performs the test loop.
        @param test_dset: The dataset to test the calibrator on.
        @return: First argument is the response confidences i.e: the mean calibrated token confidences (given in the 2nd
        argument if available). If the calibrator is not token/logit based, the 2nd argument should be a list of None.
        """
        pass

    def test(self, test_dset: DictDataset, **kwargs) -> dict:
        self.llm_bundle.lm_head.float()
        self.calibrator_model.to(DEVICE)
        self.calibrator_model.eval()

        if not self.tuned:
            warnings.warn("Calibrator model has not been loaded or trained. Expect dubious results.")

        with torch.no_grad():
            response_confs, token_confs = self.test_loop(test_dset)

        out_dict = {
            "calibrated_confs": torch.Tensor(response_confs),
            "calibrated_token_probs": token_confs,
            "calibrated_successful": torch.ones(len(test_dset)).bool()
        }
        self.llm_bundle.lm_head.half()
        return out_dict

    @abstractmethod
    def save(self, filepath, **kwargs):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


class LogitCalibrator(Calibrator, ABC):
    """
    Calibrator Class that focuses on tuning response confidences based on the logits of the responses.
    """

    @abstractmethod
    def __init__(self,
                 llm_bundle,
                 calibrator_model,
                 loss_fn: LossFunctionDetails,
                 input_key="logits",
                 label_key="correct",
                 learning_rate=None):
        super().__init__(llm_bundle, loss_fn, calibrator_model, learning_rate)

        self.label_key = label_key
        self.input_key = input_key

        self.tuned = False

    def calibration_epoch(self, calib_pbar, postfix, optimiser, **kwargs):
        postfix["total_loss_last_epoch"] = 0
        torch.autograd.set_detect_anomaly(True)

        for i, batch in enumerate(calib_pbar):
            label_batch = batch[self.label_key].to(DEVICE)
            logits_batch = batch[self.input_key].to(DEVICE).float()
            tokens_batch = batch["tokens"].to(DEVICE)

            optimiser.zero_grad()

            out_token_confs = self.calibrator_model(logits_batch, tokens_batch)
            label_batch = label_batch.to(out_token_confs.dtype)
            loss = self.loss_fn(out_token_confs, label_batch)
            loss.backward()
            for name, param in self.calibrator_model.named_parameters():
                print(f"Parameter: {name}, Gradient: {param.grad}")

            optimiser.step()
            postfix["total_loss_last_epoch"] += loss.item()

    def calibrate(self,
                  calibration_dset: DictDataset,
                  batch_size=1,
                  epochs=35,
                  _postprocess_fn=None,
                  **kwargs) -> Optional[EarlyStopping]:
        """
        Tunes the calibrator model given a dictionary dataset.
        :param calibration_dset:
        :param validation_dset:
        :param batch_size:
        :param epochs:
        :param lr:
        :param _postprocess_fn:
        :param kwargs:
        :return:
        """
        # If the model has no learnable parameters, then it is already calibrated.
        if not any(param.numel() > 0 for param in self.calibrator_model.parameters()):
            print("Model has no parameters, so no calibration performed.")
            self.tuned = True
            return None

        self.llm_bundle.load_model(silent=True, lm_head_only=True)
        self.llm_bundle.lm_head.float()

        self.calibrator_model.to(DEVICE)
        self.calibrator_model.eval()

        if _postprocess_fn is None:
            _postprocess_fn = logit_token_repeat_label_key(self.label_key, self.llm_bundle)

        calibration_dl = DataLoader(calibration_dset,
                                    collate_fn=calibration_dset.collate_fn("final_hidden_states",
                                                                           "tokens",
                                                                           self.label_key,
                                                                           postprocess_fn=_postprocess_fn),
                                    batch_size=batch_size,
                                    shuffle=True)

        # Optimise llm.
        optimiser = optim.SGD(self.calibrator_model.parameters(), lr=self.learning_rate)

        print("Training Calibrator")
        es = EarlyStopping(verbose=True)
        postfix = {}
        for epoch_idx in range(epochs):
            calib_pbar = tqdm(calibration_dl,
                              desc=f"Epoch {epoch_idx + 1}/{epochs}")

            self.calibration_epoch(calib_pbar, postfix, optimiser)
            should_stop = es(postfix["total_loss_last_epoch"], self.calibrator_model)
            if should_stop:
                print("Stopping.")
                break
            torch.cuda.empty_cache()

        es.load_checkpoint(self.calibrator_model)
        calibration_dset["calibrated_successful"] = torch.ones(len(calibration_dset)).bool()

        self.tuned = True
        self.llm_bundle.lm_head.half()
        return es

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
        response_confs_after_calib = []
        token_confs_after_calib = []
        for batch in tqdm(test_dset):
            inp = batch["final_hidden_states"].float()
            logits = self.llm_bundle.final_hs_to_logits(inp).to(DEVICE)
            tokens = batch["tokens"].to(DEVICE)
            token_confs = self.calibrator_model(logits, tokens).cpu()

            token_confs_after_calib.append(token_confs)
            response_confs_after_calib.append(torch.mean(token_confs))
        return response_confs_after_calib, token_confs_after_calib
