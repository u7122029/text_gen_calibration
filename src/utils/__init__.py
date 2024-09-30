import inspect
from abc import ABC
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Type, Any
import simple_colors as sc
from torchmetrics.classification import BinaryCalibrationError

from torch import nn

import dill
import torch

from utils.earlystopping import EarlyStopping

QUALITATIVE_SCALE = {
    "Very low": 0,
    "Low": 0.3,
    "Somewhat low": 0.45,
    "Medium": 0.5,
    "Somewhat high": 0.65,
    "High": 0.7,
    "Very high": 1,
}
RESULTS_PATH = "results"
FIGURES_PATH = "figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    with open("hf_token.txt") as f:
        HF_TOKEN = f.read().strip()
except:
    print(sc.red("hf_token.txt file containing the huggingface token not found. Some models will not load."))
    HF_TOKEN = None


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss which takes labels that transforms a label of 1 to the weight, and a label of 0 to the
    """
    def __init__(self, weight):
        super().__init__()
        assert 0 <= weight <= 1

        self.weight = weight
        self.criterion = nn.MSELoss(reduction="sum")

    def forward(self, confs, labels):
        mask = labels == 1

        correct_losses = self.criterion(confs[mask], labels[mask])
        incorrect_losses = self.criterion(confs[~mask], labels[~mask])
        return 1/len(confs) * ((1 - self.weight) * correct_losses + self.weight * incorrect_losses)


class L2ECELoss(nn.Module):
    """
    Manual implementation of the L2 ECE because the torchmetrics version cannot be backpropagated.
    """
    def __init__(self, n_bins=15, device='cuda'):
        super(L2ECELoss, self).__init__()
        self.n_bins = n_bins
        self.device = device

    def forward(self, confidences, accuracies):
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=self.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = torch.tensor(0.0, device=self.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.square(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class LossFunctionDetails:
    def __init__(self, name, loss_fn, learning_rate):
        self.name = name
        self.__loss_fn = loss_fn
        self.__learning_rate = learning_rate

    @property
    def loss_fn(self):
        return self.__loss_fn

    @property
    def learning_rate(self):
        return self.__learning_rate


class LossFunc(Enum):
    """
    Loss Function Enum
    Note that the Calibration Aware Loss is already weighted by confidence bins.
    """
    CORRECT_AWARE = 0
    BCE = 1
    WEIGHTED_CORRECT_AWARE = 2
    CALIB_AWARE = 3

    def __call__(self, *args, **kwargs):
        losses = [nn.MSELoss(),
                  nn.BCELoss(),
                  WeightedMSELoss(*args, **kwargs),
                  L2ECELoss()]
        learning_rates = [0.01, 0.001, 0.01, 0.01]
        return LossFunctionDetails(self.name, losses[self.value], learning_rates[self.value])

    @classmethod
    def from_string(cls, x):
        return cls.__members__[x]


def get_class_bases(x: Type):
    """
    Recursively obtain all superclasses of class x.
    @param x:
    @return:
    """
    bases = set()
    for base in x.__bases__:
        bases.add(base)
        bases = bases.union(get_class_bases(base))
    return bases


def class_predicate(*cls):
    def predicate_func(x):
        # Exclude functions and other non-classes.
        if not inspect.isclass(x): return False

        # Exclude classes that directly derive from ABC.
        if ABC in x.__bases__: return False

        # Check that the class x at some point derives from all classes in cls.
        class_bases = get_class_bases(x)
        return all([c in class_bases for c in cls])

    return predicate_func


def dill_load(pth: PathLike) -> Any:
    with open(pth, "rb") as f:
        out = dill.load(f)
    return out


def dill_save(obj: Any, pth: Path):
    pth.parent.mkdir(parents=True, exist_ok=True)
    with open(pth, "wb") as f:
        dill.dump(obj, f)
