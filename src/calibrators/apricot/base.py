from abc import ABC, abstractmethod

import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from torch.utils.data import DataLoader

from data import DictDataset
from utils import DEVICE


class APRICOT(ABC):
    """
    Base abstract class for the APRICOT method.
    The idea is to encode each question into latent space, clustering them by distance. Closer questions indicate that
    they are similar in some manner.
    We then take the accuracy of each question per cluster. These accuracies correspond to the target confidences that
    the responses should achieve.

    From here, this class can be extended via a child class to determine how these targets should be used.
    """
    def __init__(self):
        if type(self) is APRICOT:
            raise Exception("This class cannot be instantiated!")

    def get_target_accuracies(self, dset: DictDataset, batch_size=1, embed_model_name="all-mpnet-base-v2"):
        """

        @param dset:
        @param batch_size:
        @param embed_model_name:
        @return:
        """
        embedding_model = SentenceTransformer(embed_model_name)
        embedding_model.to(DEVICE)

        dl = DataLoader(dset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=dset.collate_fn("question", "correct"),
                        pin_memory=True)

        # Get question embeddings
        embeddings = []
        for batch in dl:
            batch = batch["question"]
            embedded_batch = embedding_model.encode(batch)
            for item in embedded_batch:
                embeddings.append(torch.Tensor(item))
        embeddings = torch.stack(embeddings, dim=0)

        # Cluster the embeddings
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=1, n_jobs=1)
        eps = 1e-6
        embeddings = ((embeddings - torch.mean(embeddings, dim=0)) /
                      (torch.std(embeddings, dim=0) + eps))  # standardise embeddings (assuming normal distribution)
        embeddings = embeddings.to(torch.float16)  # limit ram usage.
        clusterer.fit(embeddings.numpy())
        cluster_labels = clusterer.labels_

        def temp_debug(label):
            x = torch.Tensor(dset.data_dict["correct"])[cluster_labels == label].float()
            return torch.mean(x)

        label2target = {label: temp_debug(label)
                        for label in set(cluster_labels)}

        target_accuracies = []
        for i, sample in enumerate(dset):
            target_accuracies.append(label2target[cluster_labels[i]] if cluster_labels[i] > 0 else sample["correct"])
        return embeddings, torch.Tensor(target_accuracies)
