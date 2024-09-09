from enum import Enum

from torch import nn
import torch


class CrossUnifEntropy(nn.Module):
    def __init__(self, alpha=0.6):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Forward function
        @param logits: Has shape [batch, classes]
        @param labels: Has shape [batch]
        @return:
        """
        argmaxes = torch.argmax(logits, dim=1)
        correct_mask = argmaxes == labels

        ce_losses = (1 - self.alpha) * self.ce_loss(logits[correct_mask], labels[correct_mask])
        unif_losses = torch.mean(- (self.alpha / logits.shape[1]) * torch.mean(torch.log(torch.softmax(logits[~correct_mask], dim=1)),
                                                                    dim=1))
        if unif_losses.isnan():
            unif_losses = 0

        return (ce_losses + unif_losses) / 2.0


class CorrectnessAwareLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, tokens, correctness):
        prob_vecs = torch.softmax(logits, dim=1)
        confidences = torch.take_along_dim(prob_vecs, tokens.unsqueeze(1), dim=1).squeeze(1)
        return self.criterion(confidences, correctness)


class LossFunc(Enum):
    CORRECT_AWARE = 0
    CROSS_ENTROPY = 1
    CROSS_UNIF_ENTROPY = 2

    def __call__(self, *args, **kwargs):
        losses = [CorrectnessAwareLoss(nn.MSELoss()),
                  CorrectnessAwareLoss(nn.BCELoss()),
                  CrossUnifEntropy(*args, **kwargs)]
        return losses[self.value]

    @classmethod
    def from_string(cls, x):
        return cls.__members__[x]
