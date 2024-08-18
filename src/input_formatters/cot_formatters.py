import torch
from evaluate import load

from data import get_dataset, DatasetType
from prompt_formatters.cot import CoTPromptFormat, MCQCoTPromptFormat, CoTVersion, AltCoTPromptFormat, \
    AltMCQCoTPromptFormat
from .generic import CoTInputFormatter
from llm_models.textgen import TextGenLLMBundle


class GSMCoT(CoTInputFormatter):
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 cot_version: CoTVersion,
                 calib_dset_size=None,
                 test_dset_size=None):
        if cot_version == CoTVersion.ALT:
            prompt_formatter = AltCoTPromptFormat(llm_bundle)
        else:
            prompt_formatter = CoTPromptFormat(llm_bundle)

        super().__init__(llm_bundle,
                         get_dataset(DatasetType.GSM),
                         prompt_formatter,
                         calib_dset_size,
                         test_dset_size)

    def correctness(self, predictions, labels, successful):
        assert len(predictions) == len(labels)
        predictions = torch.Tensor([int(x) for x in predictions]).int()
        labels = torch.Tensor(labels)
        out = torch.zeros(len(predictions)).bool()
        out[successful] = predictions[successful] == labels[successful]
        return out.to(torch.uint8)


class MATHCoT(CoTInputFormatter):
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 cot_version: CoTVersion,
                 calib_dset_size=None,
                 test_dset_size=None):
        if cot_version == CoTVersion.ALT:
            prompt_formatter = AltCoTPromptFormat(llm_bundle)
        else:
            prompt_formatter = CoTPromptFormat(llm_bundle)

        super().__init__(llm_bundle,
                         get_dataset(DatasetType.MATH),
                         prompt_formatter,
                         calib_dset_size,
                         test_dset_size)
        self.__evl = None

    def correctness(self, predictions, labels, successful):
        assert len(predictions) == len(labels)

        if self.__evl is None:
            self.__evl = load("evaluate-metric/competition_math")

        outs = []
        for pred, label, succ in zip(predictions, labels, successful):
            if not succ:
                outs.append(0)
                continue
            outs.append(self.__evl.compute(references=[label], predictions=[pred])["accuracy"])

        return torch.Tensor(outs).to(torch.uint8)


class AQUARATCoT(CoTInputFormatter):
    def __init__(self, llm_bundle: TextGenLLMBundle, cot_version: CoTVersion, calib_dset_size=None, test_dset_size=None):
        if cot_version == CoTVersion.ALT:
            prompt_formatter = AltMCQCoTPromptFormat(llm_bundle)
        else:
            prompt_formatter = MCQCoTPromptFormat(llm_bundle)
        super().__init__(llm_bundle,
                         get_dataset(DatasetType.MATH),
                         prompt_formatter,
                         calib_dset_size,
                         test_dset_size)

    def correctness(self, predictions, labels, successful):
        correctness = []
        for pred, label, succ in zip(predictions, labels, successful):
            if not succ:
                correctness.append(False)
                continue
            pred = pred.upper()
            correctness.append(pred == label)
        return torch.Tensor(correctness).to(torch.uint8)


