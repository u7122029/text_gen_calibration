import torch
from evaluate import load

from data import get_dataset, DatasetType
from .generic import CoTInputFormatter
from llm_models import TextGenLLMBundle
from prompt_formatters import CoTPromptFormat, MCQCoTPromptFormat


class GSMCoT(CoTInputFormatter):
    def __init__(self, llm_bundle: TextGenLLMBundle, calib_dset_size=None, test_dset_size=None):
        super().__init__(llm_bundle,
                         get_dataset(DatasetType.GSM),
                         CoTPromptFormat(llm_bundle),
                         calib_dset_size,
                         test_dset_size)

    def correctness(self, predictions, labels):
        predictions = [int(x) for x in predictions]
        return (torch.Tensor(predictions).int() == torch.Tensor(labels)).to(torch.uint8)


class MATHCoT(CoTInputFormatter):
    def __init__(self, llm_bundle: TextGenLLMBundle, calib_dset_size=None, test_dset_size=None):
        super().__init__(llm_bundle,
                         get_dataset(DatasetType.MATH),
                         CoTPromptFormat(llm_bundle),
                         calib_dset_size,
                         test_dset_size)
        self.__evl = None

    def correctness(self, predictions, labels):
        assert len(predictions) == len(labels)

        if self.__evl is None:
            self.__evl = load("evaluate-metric/competition_math")

        outs = []
        for pred, label in zip(predictions, labels):
            outs.append(self.__evl.compute(references=[label], predictions=[pred])["accuracy"])

        return torch.Tensor(outs).to(torch.uint8)


class AQUARATCoT(CoTInputFormatter):
    """
    TODO: FINISH THIS CLASS!
    """
    def __init__(self, llm_bundle: TextGenLLMBundle, calib_dset_size=None, test_dset_size=None):
        super().__init__(llm_bundle,
                         get_dataset(DatasetType.MATH),
                         MCQCoTPromptFormat(llm_bundle),
                         calib_dset_size,
                         test_dset_size)
        self.__evl = None


