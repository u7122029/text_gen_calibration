import warnings
from abc import ABC

import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import DictDataset
from utils import DEVICE, LLMBundle, dill_load, dill_save
from .generic import Calibrator


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
            embedded_batch = embedding_model.encode(batch)
            for item in embedded_batch:
                embeddings.append(torch.Tensor(item))
        embeddings = torch.stack(embeddings, dim=0)
        print(embeddings.shape)

        # Cluster the embeddings
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=1, n_jobs=1)
        eps = 1e-6
        embeddings = ((embeddings - torch.mean(embeddings, dim=0)) /
                      (torch.std(embeddings, dim=0) + eps))  # standardise embeddings (assuming normal distribution)
        embeddings = embeddings.to(torch.float16)  # limit ram usage.
        clusterer.fit(embeddings.numpy())
        cluster_labels = clusterer.labels_

        def temp_debug(label):
            x = torch.Tensor(calibration_dset.data_dict["correct"])[cluster_labels == label].float()
            return torch.mean(x)

        label2target = {label: temp_debug(label)
                        for label in set(cluster_labels)}

        target_accuracies = []
        for i, sample in enumerate(calibration_dset):
            target_accuracies.append(label2target[cluster_labels[i]] if cluster_labels[i] > 0 else sample["correct"])
        return embeddings, torch.Tensor(target_accuracies)

    def __collate_post_process(self, out_dict):
        #out_dict["correct"] = torch.cat(out_dict["correct"]).float()
        return out_dict


class APRICOT_Original(APRICOT):
    def __init__(self, llm_bundle: LLMBundle, calib_model_name: str="microsoft/deberta-v3-base"):
        super().__init__(llm_bundle)
        self.calibrator_model = AutoModelForSequenceClassification.from_pretrained(calib_model_name)
        self.calibrator_tokeniser = AutoTokenizer.from_pretrained(calib_model_name)
        self.tuned = False

    def calibrate(self, calibration_dset: DictDataset, batch_size=1, epochs=30, **kwargs):
        embeddings, target_accuracies = self.get_target_accuracies(calibration_dset, batch_size)
        #calibration_dset.add_column("question_embeds", embeddings)
        calibration_dset.add_column("target_confs", target_accuracies)

        def temp_postprocess(out_dict):
            out_dict["target_confs"] = torch.Tensor(out_dict["target_confs"])
            return out_dict

        dl = DataLoader(calibration_dset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=calibration_dset.collate_fn("tokens", "target_confs",
                                                               postprocess_fn=temp_postprocess))

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
                encoded = self.calibrator_tokeniser(self.llm_bundle.tokeniser.batch_decode(batch["tokens"]),
                                                    return_tensors="pt",
                                                    padding=True).to(DEVICE)
                labels = batch["target_confs"].to(DEVICE)

                optimiser.zero_grad()
                outs = self.calibrator_model(**encoded)
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
        print(len(confs_after_calibration))
        return confs_after_calibration

    def test(self, test_dset: DictDataset, **kwargs):
        if not self.tuned:
            warnings.warn("Calibrator model has not been loaded or trained. Expect dubious results.")

        self.calibrator_model = self.calibrator_model.to(DEVICE)
        with torch.no_grad():
            confs_after_calibration = self.test_loop(test_dset)
        return torch.Tensor(confs_after_calibration)