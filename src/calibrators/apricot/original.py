from data import DictDataset
from llm_models.generic import LLMBundle
from .base import APRICOT
from ..token_response_scaler import TokenCalibrator


class APRICOT_Original(APRICOT, TokenCalibrator):
    """
    Uses the APRICOT method to determine the target confidences for each question in the calibration dataset.
    Then we train an LLM for sequence classification to ensure that each tokenised question with response attains a
    confidence that is as close as possible to these targets.
    This method of optimisation corresponds with the original APRICOT method proposed in the respective paper.
    """
    def __init__(self, llm_bundle: LLMBundle):
        APRICOT.__init__(self, llm_bundle)
        TokenCalibrator.__init__(self, llm_bundle, label_key="target_confs")

    def calibrate(self, calibration_dset: DictDataset, batch_size=1, epochs=30, lr=1e-3, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, batch_size)
        calibration_dset["target_confs"] = target_accuracies

        TokenCalibrator.calibrate(self, calibration_dset, batch_size, epochs, lr, **kwargs)
        """dl = DataLoader(calibration_dset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=calibration_dset.collate_fn("question", "tokens", "target_confs",
                                                               postprocess_fn=postprocess_target_confs))

        optimiser = optim.SGD(self.calibrator_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss().to(DEVICE)
        
        self.calibrator_model.to(DEVICE)
        self.calibrator_model.train()

        es = EarlyStopping(verbose=True)
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

            should_stop = es(postfix["total_loss_last_epoch"], self.calibrator_model)
            if should_stop:
                break

        es.load_checkpoint(self.calibrator_model)
        calibration_dset["calibrated_successful"] = torch.ones(len(calibration_dset)).bool()

        self.calibrator_model.eval()
        self.tuned = True"""
