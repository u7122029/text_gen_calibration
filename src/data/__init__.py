from enum import Enum
from torch.utils.data import DataLoader

from data.dictdataset import DictDataset
from data.gsm import get_gsm
from data.math_dset import get_math
from data.aqua_rat import get_aqua_rat
from data.trivia_qa import get_trivia_qa


class DatasetType(Enum):
    GSM = 0
    MATH = 1
    AQUARAT = 2
    TRIVIAQA = 3


def get_dataset(name: DatasetType) -> DictDataset:
    assert isinstance(name, DatasetType), f"{name} is not a dataset type."
    out = None
    if name == DatasetType.GSM:
        out = get_gsm()
    elif name == DatasetType.MATH:
        out = get_math()
    elif name == DatasetType.AQUARAT:
        out = get_aqua_rat()
    elif name == DatasetType.TRIVIAQA:
        out = get_trivia_qa()
    return out


if __name__ == "__main__":
    import torch
    x = DictDataset({"a": [1, 2, 3, 4, 5, 6], "b": torch.rand(6)})

    dl = DataLoader(x, batch_size=4)
    print(next(iter(dl)))