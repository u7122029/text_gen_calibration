from data import get_dataset, DatasetType
from .generic import CoTInputFormatter
from utils import TextGenLLMBundle


class GSMCoT(CoTInputFormatter):
    def __init__(self, llm_bundle: TextGenLLMBundle, calib_dset_size, test_dset_size=None):
        super().__init__(llm_bundle, get_dataset(DatasetType.GSM), calib_dset_size, test_dset_size)


class MATHCoT(CoTInputFormatter):
    def __init__(self, llm_bundle: TextGenLLMBundle, calib_dset_size, test_dset_size=None):
        super().__init__(llm_bundle, get_dataset(DatasetType.MATH), calib_dset_size, test_dset_size)




