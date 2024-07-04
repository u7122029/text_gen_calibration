from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from pathlib import Path

RESULTS_PATH = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AbsModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class TokenLogitsDataset(Dataset):
    def __init__(self, logits, tokens, correct):
        """

        :param logits: List of tensors. The length of the list is the number of responses, and the shape of each tensor
        is [response_length (num_tokens), vocab_size]
        :param tokens: List of tensors. The length of the list is the number of responses, and the shape of each tensor
        is [response_length (num_tokens)]
        :param correct: Tensor involving boolean values of shape [num_responses].
        """
        self.logits = logits
        self.tokens = tokens
        self.correct = correct

        self.vocab_size = self.logits[0].shape[1]

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


class TLTokenFrequencyDataset(TokenLogitsDataset):
    #def __init__(self, logits, tokens, correct):
    #    super().__init__(logits, tokens, correct)
    # num_responses = len(self.tokens)

    # Construct term-document frequency matrix
    """term_doc_freqs = torch.zeros(num_responses, self.vocab_size, dtype=torch.uint16)
    for response_idx, token_response in enumerate(self.tokens):

        token_counts = torch.bincount(token_response)
        term_doc_freqs[response_idx, :len(token_counts)] += token_counts

    self.relative_tfs = (term_doc_freqs / torch.sum(term_doc_freqs, dim=1, keepdim=True))"""

    def __getitem__(self, item):
        tokens = self.tokens[item]
        relative_tfs = torch.zeros(self.vocab_size)
        token_counts = torch.bincount(tokens)
        relative_tfs[:len(token_counts)] += token_counts
        relative_tfs /= torch.sum(relative_tfs)
        return self.logits[item], tokens, self.correct_vectors[item], relative_tfs

    @staticmethod
    def collate_fn(data):
        logits = []
        tokens = []
        correct_vecs = []
        relative_tfs = []
        for x in data:
            logits.append(x[0])
            tokens.append(x[1])
            correct_vecs.append(x[2])
            relative_tfs.append(x[3])
        return logits, tokens, torch.cat(correct_vecs), torch.stack(relative_tfs)


