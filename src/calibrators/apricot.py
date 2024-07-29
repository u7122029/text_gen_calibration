from typing import List

from torch.utils.data import DataLoader

from data import DictDataset
from generic import Calibrator
from sentence_transformers import SentenceTransformer
import torch
from utils import DEVICE
from sklearn.cluster import HDBSCAN


class APRICOT(Calibrator):
    def calibrate(self, calibration_dset: DictDataset, batch_size=1, **kwargs):
        embedding_model = SentenceTransformer("all-mpnet-base-v2")
        embedding_model.to(DEVICE)
        calibration_dset.restrict_keys(self.dset_columns())
        dl = DataLoader(calibration_dset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        # Get question embeddings
        embeddings = []
        for batch in dl:
            batch = batch.to(DEVICE)
            embedded_batch = embedding_model(batch)
            for item in embedded_batch:
                embeddings.append(item)
        embeddings = torch.cat(embeddings, dim=0)

        # Cluster the embeddings
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=1, n_jobs=1)
        eps = 1e-6
        embeddings = (embeddings - torch.mean(embeddings, dim=0)) / (torch.std(embeddings, dim=0) + eps)  # standardise embeddings (assuming normal distribution)
        embeddings = embeddings.astype(torch.float16)  # To not blow up system memory
        clusterer.fit(embeddings.numpy())
        cluster_labels = clusterer.labels_
        label2target = {label: torch.mean(torch.Tensor(calibration_dset["correct"])[cluster_labels == label])
                        for label in set(cluster_labels)}
        question_id_to_targets = {
            question_id: label2target[cluster_labels[i]] if cluster_labels[i] > 0 else accuracies[i]
            for i, question_id in enumerate(question_ids)
        } # TODO: FIX THIS!
        calibration_dset.reset_keys()

    def collate_fn(self, data_list):
        out = {
            "question": [],
            "correct": []
        }
        for d in data_list:
            out["question"].append(d["question"])
            #out["tokens"].append(d["tokens"])
            out["correct"].append(d["correct"])

        out["correct"] = torch.cat(out["correct"]).float()
        return out

    def dset_columns(self) -> List[str]:
        return ["question", "correct"]

    def load(self, filepath):
        pass

    def save(self, filepath, **kwargs):
        pass

    def test(self, test_dset: DictDataset, **kwargs):
        pass