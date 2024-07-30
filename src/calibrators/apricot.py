from typing import List

from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from data import DictDataset
from generic import Calibrator
from sentence_transformers import SentenceTransformer
import torch
from utils import DEVICE, LLMBundle
from sklearn.cluster import HDBSCAN
from abc import ABC
from tqdm import tqdm


class APRICOT(Calibrator, ABC):
    def get_target_accuracies(self, calibration_dset: DictDataset, batch_size=1, embed_model_name="all-mpnet-base-v2"):
        """

        @param calibration_dset:
        @param batch_size:
        @param embed_model_name:
        @return:
        """
        embedding_model = SentenceTransformer(embed_model_name)
        embedding_model.to(DEVICE)

        dl = DataLoader(calibration_dset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=calibration_dset.collate_fn("question",
                                                               "correct",
                                                               postprocess_fn=self.__collate_post_process))

        # Get question embeddings
        embeddings = []
        for batch in dl:
            batch = batch["question"]
            embedded_batch = embedding_model(batch)
            for item in embedded_batch:
                embeddings.append(item)
        embeddings = torch.cat(embeddings, dim=0)

        # Cluster the embeddings
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=1, n_jobs=1)
        eps = 1e-6
        embeddings = ((embeddings - torch.mean(embeddings, dim=0)) /
                      (torch.std(embeddings, dim=0) + eps))  # standardise embeddings (assuming normal distribution)
        embeddings = embeddings.astype(torch.float16)  # limit ram usage.
        clusterer.fit(embeddings.numpy())
        cluster_labels = clusterer.labels_
        label2target = {label: torch.mean(torch.Tensor(calibration_dset["correct"])[cluster_labels == label])
                        for label in set(cluster_labels)}

        target_accuracies = []
        for i, sample in enumerate(calibration_dset):
            target_accuracies.append(label2target[cluster_labels[i]] if cluster_labels[i] > 0 else sample["correct"])
        return embeddings, torch.Tensor(target_accuracies)

    def __collate_post_process(self, out_dict):
        out_dict["correct"] = torch.cat(out_dict["correct"]).float()
        return out_dict


class APRICOT_Original(APRICOT):
    def __init__(self, llm_bundle: LLMBundle, calib_model_name: str):
        super().__init__(llm_bundle)
        self.calibrator_model = AutoModelForSequenceClassification.from_pretrained(calib_model_name)
        self.tuned = False

    def calibrate(self, calibration_dset: DictDataset, batch_size=1, epochs=30, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, batch_size)
        #calibration_dset.add_column("question_embeds", embeddings)
        calibration_dset.add_column("target_confs", target_accuracies)

        dl = DataLoader(calibration_dset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=calibration_dset.collate_fn("tokens", "target_confs"))

        optimiser = optim.SGD(self.calibrator_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss().to(DEVICE)
        self.calibrator_model.to(DEVICE)
        self.calibrator_model.train()

        postfix = {}
        for epoch_idx in range(epochs):
            pbar = tqdm(dl,
                        desc=f"Epoch {epoch_idx + 1}/{epochs}",
                        postfix=postfix)
            for batch in pbar:
                tokens = batch["tokens"]
                labels = batch["target_confs"].to(DEVICE)

                optimiser.zero_grad()
                input_ids = torch.Tensor(self.llm_bundle.tokeniser.pad_sequences(tokens, padding="longest"))
                attention_mask = input_ids != self.llm_bundle.tokeniser.pad_token_id

                outs = self.calibrator_model(input_ids=input_ids, attention_mask=attention_mask)
                confs = torch.softmax(outs.logits, dim=-1)[:, 1]
                loss = criterion(confs, labels)
                loss.backward()
                optimiser.step()
                postfix["total_loss_last_epoch"] += loss.item()

        self.calibrator_model.eval()
        self.tuned = True

    def load(self, filepath):
        pass

    def save(self, filepath, **kwargs):
        pass

    def test(self, test_dset: DictDataset, **kwargs):
        pass