from enum import Enum

from data.dictdataset import DictDataset
from data.gsm import get_gsm
from data.math_dset import get_math
from data.aqua_rat import get_aqua_rat
from data.squad_v2 import get_squad_v2


class DatasetType(Enum):
    GSM = 0
    MATH = 1
    AQUARAT = 2
    SQUADV2 = 3

    def __call__(self) -> DictDataset:
        funcs = [get_gsm, get_math, get_aqua_rat, get_squad_v2]
        return funcs[self.value]()
