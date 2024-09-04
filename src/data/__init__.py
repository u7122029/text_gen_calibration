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

    def __call__(self) -> DictDataset:
        funcs = [get_gsm, get_math, get_aqua_rat, get_trivia_qa]
        return funcs[self.value]()
