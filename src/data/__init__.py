from enum import Enum

from data.dictdataset import DictDataset
from data.gsm import get_gsm
from data.math_dset import get_math
from torch.utils.data import DataLoader


class DatasetType(Enum):
    GSM = 0
    MATH = 1


def get_dataset(name: DatasetType) -> DictDataset:
    assert isinstance(name, DatasetType), f"{name} is not a dataset type."
    out = None
    if name == DatasetType.GSM:
        out = get_gsm()
    elif name == DatasetType.MATH:
        out = get_math()
    return out


if __name__ == "__main__":
    import torch
    x = DictDataset({"a": [1, 2, 3, 4, 5, 6], "b": torch.rand(6)})

    dl = DataLoader(x, batch_size=4)
    print(next(iter(dl)))