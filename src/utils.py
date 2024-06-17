from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from pathlib import Path

RESULTS_PATH = "results"


class TokenLogitsDataset(Dataset):
    def __init__(self, logits, tokens, correct):
        self.logits = logits
        self.tokens = tokens
        self.correct = correct

        assert len(self.logits) == len(self.tokens), \
            f"given logits is not the same length as the tokens. len(logits): {len(self.logits)}, len(tokens): {len(self.tokens)}"
        assert len(self.tokens) == len(self.correct), \
            f"given tokens is not the same length as the labels. len(tokens): {len(self.tokens)}, len(correct): {len(self.correct)}."

        self.correct_vectors = []
        for t, c in zip(self.tokens, self.correct):
            vec = torch.zeros(len(t)) + c
            self.correct_vectors.append(vec)

    def __getitem__(self, item):
        return self.logits[item], self.tokens[item], self.correct_vectors[item]

    def __len__(self):
        return len(self.correct)

    @staticmethod
    def collate_fn(data):
        logits = []
        tokens = []
        correct_vecs = []
        for x in data:
            logits.append(x[0])
            tokens.append(x[1])
            correct_vecs.append(x[2])
        return torch.cat(logits), torch.cat(tokens), torch.cat(correct_vecs)
