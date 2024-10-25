from typing import Type

import torch
from evaluate import load

from calibrators import Calibrator
from data import DatasetType
from llm_models.textgen import TextGenLLMBundle
from prompt_formatters.cot import PromptVersion
from utils import LossFunc
from .generic import CoTInputFormatter


class GSMCoT(CoTInputFormatter):
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 prompt_version: PromptVersion,
                 calibrator_type: Type[Calibrator],
                 loss_fn: LossFunc,
                 calib_dset_size=None,
                 test_dset_size=None):
        super().__init__(llm_bundle,
                         DatasetType.GSM(),
                         prompt_version,
                         calibrator_type,
                         loss_fn,
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
                 prompt_version: PromptVersion,
                 calibrator_type: Type[Calibrator],
                 loss_fn: LossFunc,
                 calib_dset_size=None,
                 test_dset_size=None):
        super().__init__(llm_bundle,
                         DatasetType.MATH(),
                         prompt_version,
                         calibrator_type,
                         loss_fn,
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
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 prompt_version: PromptVersion,
                 calibrator_type: Type[Calibrator],
                 loss_fn: LossFunc,
                 calib_dset_size=None,
                 test_dset_size=None):
        super().__init__(llm_bundle,
                         DatasetType.AQUARAT(),
                         prompt_version,
                         calibrator_type,
                         loss_fn,
                         calib_dset_size,
                         test_dset_size,
                         _pf_variant="mcq",
                         _mcq_options={"a", "b", "c", "d", "e"})

    def correctness(self, predictions: list[str], labels: list[str], successful: torch.Tensor):
        assert len(predictions) == len(labels)
        correctness = []
        for pred, label, succ in zip(predictions, labels, successful):
            if not succ:
                correctness.append(False)
                continue
            pred = pred.upper()
            correctness.append(pred == label)
        return torch.Tensor(correctness).to(torch.uint8)


class MMLUCoT(CoTInputFormatter):
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 prompt_version: PromptVersion,
                 calibrator_type: Type[Calibrator],
                 loss_fn: LossFunc,
                 calib_dset_size=None,
                 test_dset_size=None):
        super().__init__(llm_bundle,
                         DatasetType.MMLU(),
                         prompt_version,
                         calibrator_type,
                         loss_fn,
                         calib_dset_size,
                         test_dset_size,
                         _pf_variant="mcq",
                         _mcq_options={"a", "b", "c", "d"})

    def correctness(self, predictions: list[str], labels: list[str], successful: torch.Tensor):
        assert len(predictions) == len(labels)
        correctness = []
        for pred, label, succ in zip(predictions, labels, successful):
            if not succ:
                correctness.append(False)
                continue
            pred = pred.upper()
            correctness.append(pred == label)
        return torch.Tensor(correctness).to(torch.uint8)


class SQUADV2CoT(CoTInputFormatter):
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 prompt_version: PromptVersion,
                 calibrator_type: Type[Calibrator],
                 loss_fn: LossFunc,
                 calib_dset_size=None,
                 test_dset_size=None):
        super().__init__(llm_bundle,
                         DatasetType.SQUADV2(),
                         prompt_version,
                         calibrator_type,
                         loss_fn,
                         calib_dset_size,
                         test_dset_size,
                         _pf_variant="worded")

    def correctness(self, predictions: list[str], labels: list, successful: torch.Tensor):
        assert len(predictions) == len(labels)
        metric = load("squad_v2")

        correctness = []
        for pred, label, succ in zip(predictions, labels, successful):
            if not succ:
                correctness.append(False)
                continue
            prediction = [{"prediction_text": pred, "id": "x", "no_answer_probability": 0}]
            reference = [{"answers": label, "id": "x"}]
            result = metric.compute(predictions=prediction, references=reference)["f1"]
            correctness.append(result >= 50)
        return torch.Tensor(correctness).to(torch.uint8)

