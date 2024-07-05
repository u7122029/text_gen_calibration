from abc import ABC, abstractmethod
from utils import DEVICE, TokenLogitsDataset
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
    def __init__(self, llm_bundle):
        self.llm_bundle = llm_bundle
        self.calibrator_model = None

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def calibrate(self, **kwargs):
        pass

    @abstractmethod
    def test(self, **kwargs):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
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
        for logits_batch, _, is_correct_batch in pbar:
            logits_batch = logits_batch.to(DEVICE)
            is_correct_batch = is_correct_batch.to(DEVICE)

            optimiser.zero_grad()
            out_token_confs = self.calibrator_model(logits_batch)
            loss = loss_fn(out_token_confs, is_correct_batch)
            loss.backward()
            optimiser.step()
            postfix["total_loss_last_epoch"] += loss.item()

    def calibrate(self,
                  calib_tokens,
                  calib_logits,
                  correct,
                  batch_size,
                  epochs=30,
                  lr=0.01,
                  **kwargs):
        """
        Calibrates the calibrator model. By default, this will use the TokenLogitsDataset. You will need to override
        this function if you want to use a different dataset.
        :param calib_tokens:
        :param calib_logits:
        :param correct:
        :param batch_size:
        :param epochs:
        :param lr:
        :param kwargs:
        :return:
        """
        # Assume calib_logits has shape [dset_length, response_length, vocab_size]
        calibration_dset = self.get_dataset(calib_tokens, calib_logits, correct)
        calibration_dl = DataLoader(calibration_dset,
                                    batch_size=batch_size,
                                    collate_fn=calibration_dset.__class__.collate_fn,
                                    shuffle=True)
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

    def load(self, filepath):
        self.calibrator_model.load_state_dict(torch.load(filepath))
        self.calibrator_model.eval()

    def save(self, filepath):
        torch.save(self.calibrator_model.state_dict(), filepath)

    def test_loop(self, test_dset):
        confs_after_calibration = []
        for logits, tokens, _ in tqdm(test_dset):
            logits = logits.to(DEVICE)
            tokens = tokens.to(DEVICE)
            token_confs = self.calibrator_model(logits, tokens).cpu()
            out = torch.mean(token_confs)
            confs_after_calibration.append(out)
        return confs_after_calibration

    def test(self, test_tokens, test_logits, correct, **kwargs):
        print(test_tokens[0].shape)
        print(test_logits[0].shape)
        if not self.tuned:
            warnings.warn("Calibrator model has not been loaded or trained. Expect dubious results.")
        test_dset = self.get_dataset(test_tokens, test_logits, correct)

        self.calibrator_model = self.calibrator_model.to(DEVICE)
        with torch.no_grad():
            confs_after_calibration = self.test_loop(test_dset)
        return torch.Tensor(confs_after_calibration)