import warnings

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import APRICOT
from collate_postprocess_functions import postprocess_target_confs
from data import DictDataset
from utils import LLMBundle, DEVICE, dill_load, dill_save


class APRICOT_Original(APRICOT): # TODO: MAYBE THIS SHOULD BE A CHILD CLASS OF TokenCalibrator AND APRICOT.
    """
    Uses the APRICOT method to determine the target confidences for each question in the calibration dataset.
    Then we train an LLM for sequence classification to ensure that each tokenised question with response attains a
    confidence that is as close as possible to these targets.
    This method of optimisation corresponds with the original APRICOT method proposed in the respective paper.
    """
    def __init__(self, llm_bundle: LLMBundle, calib_model_name: str="microsoft/deberta-v3-base"):
        super().__init__(llm_bundle)
        self.calibrator_model = AutoModelForSequenceClassification.from_pretrained(calib_model_name)
        self.calibrator_tokeniser = AutoTokenizer.from_pretrained(calib_model_name)
        self.tuned = False

    def calibrate(self, calibration_dset: DictDataset, batch_size=1, epochs=30, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, batch_size)
        calibration_dset["target_confs"] = target_accuracies

        dl = DataLoader(calibration_dset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=calibration_dset.collate_fn("question", "tokens", "target_confs",
                                                               postprocess_fn=postprocess_target_confs))

        optimiser = optim.SGD(self.calibrator_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss().to(DEVICE)
        self.calibrator_model.to(DEVICE)
        self.calibrator_model.train()

        postfix = {}
        for epoch_idx in range(epochs):
            pbar = tqdm(dl,
                        desc=f"Epoch {epoch_idx + 1}/{epochs}",
                        postfix=postfix)
            postfix["total_loss_last_epoch"] = 0
            for batch in pbar:
                decoded_responses = self.llm_bundle.tokeniser.batch_decode(batch["tokens"])
                questions_and_answers = [f"{q}\n{a}" for q, a in zip(batch["question"], decoded_responses)]
                encoded_responses = self.calibrator_tokeniser(questions_and_answers,
                                                              return_tensors="pt",
                                                              padding=True).to(DEVICE)
                labels = batch["target_confs"].to(DEVICE)

                optimiser.zero_grad()
                outs = self.calibrator_model(**encoded_responses)
                confs = torch.softmax(outs.logits, dim=-1)[:, 1]
                loss = criterion(confs, labels)
                loss.backward()
                optimiser.step()
                postfix["total_loss_last_epoch"] += loss.item()

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
            encoded = self.calibrator_tokeniser(self.llm_bundle.tokeniser.decode(batch["tokens"]),
                                                return_tensors="pt",
                                                padding=True).to(DEVICE)
            outs = self.calibrator_model(**encoded)
            out = torch.softmax(outs.logits, dim=-1)[:, 1]
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
            "calibrated_successful": torch.ones(len(confs_after_calibration)).bool()
        }
        return out_dict
