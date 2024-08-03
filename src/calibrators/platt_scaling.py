import warnings

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from data import DictDataset
from utils import DEVICE, dill_save, dill_load
from .generic import LogitCalibrator, Calibrator
from .universal_calibration_models import PlattScalerLogits, PlattScalerConfs


class LogitConfsPlattScaling(LogitCalibrator):
    """
    For a single logit matrix, this calibrator takes the softmax, then the corresponding token confidences.
    The calibrator will use platt scaling on each of these confidences before taking the mean of all of them which gives
    the overall response confidence.
    """
    def __init__(self, llm_bundle):
        super().__init__(llm_bundle, PlattScalerLogits())


class MeanLogitConfsPlattScaling(LogitCalibrator):
    """
    Performs platt scaling after the mean token confidence has been taken for the response.
    """
    def __init__(self, llm_bundle):
        super().__init__(llm_bundle, PlattScalerConfs())

    def calibration_epoch(self, pbar, postfix, optimiser, **kwargs):
        postfix["total_loss_last_epoch"] = 0
        for batch in pbar:
            label_batch = torch.Tensor(batch[self.label_key]).to(DEVICE)
            logits_batch = [x.to(DEVICE) for x in batch["logits"]]
            tokens_batch = [x.to(DEVICE) for x in batch["tokens"]]

            optimiser.zero_grad()
            out_token_confs = self.calibrator_model(logits_batch, tokens_batch)
            loss = self.loss_fn(out_token_confs, label_batch)
            loss.backward()
            optimiser.step()
            postfix["total_loss_last_epoch"] += loss.item()

    def calibrate(self, *args, **kwargs):
        kwargs["_postprocess_fn"] = lambda x: x
        super().calibrate(*args, **kwargs)

    def test_loop(self, test_dset):
        confs_after_calibration = []
        for batch in tqdm(test_dset):
            logits = [batch["logits"].to(DEVICE)]
            tokens = [batch["tokens"].to(DEVICE)]
            out = self.calibrator_model(logits, tokens).cpu()
            confs_after_calibration.extend(out)
        return confs_after_calibration


class VCPlattScaling(Calibrator):
    def __init__(self, llm_bundle):
        super().__init__(llm_bundle)
        self.calibrator_model = PlattScalerConfs()
        self.loss_fn = nn.MSELoss()

    def __get_input_confs_and_correctness(self, batch):
        batch = {k: torch.Tensor(v) for k, v in batch.items()}
        numeric_confs = batch["numeric_confs"]
        worded_confs = batch["worded_confs"]
        numeric_successful = batch["numeric_successful"].bool()
        worded_successful = batch["worded_successful"].bool()
        correct = batch["correct"]

        all_confs = torch.zeros(len(numeric_confs))
        all_confs[numeric_successful] = numeric_confs[numeric_successful]
        all_confs[worded_successful & ~numeric_successful] = worded_confs[worded_successful & ~numeric_successful]
        mask = numeric_successful | worded_successful

        correct = correct[mask]
        input_confs = all_confs[mask]

        return input_confs, correct, mask

    def calibration_epoch(self, pbar, postfix, optimiser, **kwargs):
        postfix["total_loss_last_epoch"] = 0
        for batch in pbar:
            input_confs, correct, _ = self.__get_input_confs_and_correctness(batch)
            input_confs = input_confs.to(DEVICE)
            correct = correct.to(DEVICE)

            optimiser.zero_grad()
            out_token_confs = self.calibrator_model(input_confs.unsqueeze(1))
            loss = self.loss_fn(out_token_confs, correct)
            loss.backward()
            optimiser.step()
            postfix["total_loss_last_epoch"] += loss.item()

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
        calibration_dl = DataLoader(calibration_dset,
                                    collate_fn=calibration_dset.collate_fn("numeric_confs",
                                                                           "numeric_successful",
                                                                           "worded_confs",
                                                                           "worded_successful",
                                                                           "correct"),
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

    def test_loop(self, test_dl):
        confs_after_calibration = []
        all_successfuls = []
        for batch in tqdm(test_dl):
            input_confs, _, successful = self.__get_input_confs_and_correctness(batch)
            input_confs = input_confs.to(DEVICE)

            out_token_confs = self.calibrator_model(input_confs.unsqueeze(1)).cpu()

            processed = torch.zeros(len(successful))
            processed[successful] = out_token_confs
            confs_after_calibration.append(processed)
            all_successfuls.append(successful)
        return torch.cat(confs_after_calibration), torch.cat(all_successfuls)

    def test(self, test_dset: DictDataset, batch_size=1, **kwargs):
        if not self.tuned:
            warnings.warn("Calibrator model has not been loaded or trained. Expect dubious results.")

        self.calibrator_model = self.calibrator_model.to(DEVICE)
        test_dl = DataLoader(test_dset,
                             collate_fn=test_dset.collate_fn("numeric_confs",
                                                             "numeric_successful",
                                                             "worded_confs",
                                                             "worded_successful",
                                                             "correct"),
                             batch_size=batch_size)

        with torch.no_grad():
            confs_after_calibration, successful_confs = self.test_loop(test_dl)

        out_dict = {
            "calibrated_confs": torch.Tensor(confs_after_calibration),
            "calibrated_successful": torch.Tensor(successful_confs)
        }
        return out_dict
