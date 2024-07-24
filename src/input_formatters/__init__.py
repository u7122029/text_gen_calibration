import torch
from torch import nn
import re
from datasets import Dataset
from enum import Enum
from torch.utils.data import DataLoader
from pathlib import Path

from data import get_dataset, DictDataset
from utils import (RESULTS_PATH,
                   TextGenLLMBundle,
                   class_predicate,
                   COT_SYSTEM_PROMPT,
                   WORDED_CONF_PROMPT,
                   NUMERIC_CONF_PROMPT,
                   FINAL_ANSWER_FORMAT,
                   QUESTION_FORMAT)
from calibrators import Calibrator
from typing import Type, Optional, Tuple
import inspect
import sys
from abc import ABC, abstractmethod


class CoTFormat(Enum):
    SYSTEM_USER_CHAT = 0
    USER_CHAT = 1
    NO_TEMPLATE = 2

    @classmethod
    def from_model_name(cls, name):
        name_dict = {
            "google/gemma-1.1-2b-it": cls.USER_CHAT,
            "google/gemma-1.1-7b-it": cls.USER_CHAT,
            "google/gemma-2-9b-it": cls.USER_CHAT,
            "HuggingFaceH4/zephyr-7b-beta": cls.SYSTEM_USER_CHAT,
            "meta-llama/Meta-Llama-3-8B-Instruct": cls.SYSTEM_USER_CHAT,
            "mistralai/Mistral-7B-Instruct-v0.3": cls.USER_CHAT,
            "01-ai/Yi-1.5-9B-Chat": cls.SYSTEM_USER_CHAT,
            "NousResearch/Hermes-2-Theta-Llama-3-8B": cls.SYSTEM_USER_CHAT,
            "NousResearch/Hermes-2-Pro-Mistral-7B": cls.SYSTEM_USER_CHAT,
            "microsoft/Phi-3-small-128k-instruct": cls.USER_CHAT,
            "microsoft/Phi-3-mini-128k-instruct": cls.USER_CHAT,
            "microsoft/Phi-3-mini-4k-instruct": cls.USER_CHAT
        }
        return name_dict[name]


class InputFormatter(ABC):
    """
    TODO: Determine methods that should be common across all subclasses.
    """

    @abstractmethod
    def __init__(self):
        """
        Abstract constructor to ensure that this class cannot be instantiated.
        """
        pass


class GSMCoT(InputFormatter):
    """
    The idea is that we will ask for the model's answer, and also ask for the model's verbalised confidence.
    Both qualitative and quantitative
    """

    def __init__(self, llm_bundle: TextGenLLMBundle, calib_dset_size, test_dset_size=None):
        """

        :param llm_bundle:
        :param calib_dset_size: Calibration set size.
        :param test_dset_size: Test set size (if None, uses the rest of the dataset)
        """
        self.llm_bundle = llm_bundle
        self.task_name = "GSM"
        self.dataset = get_dataset("GSM")
        self.__calibrator: Optional[Calibrator] = None

        self.target_dir = Path(RESULTS_PATH) / self.llm_bundle.llm_name / self.task_name
        self.target_dir.mkdir(parents=True, exist_ok=True)

        indices = torch.randperm(len(self.dataset))
        calib_indices = indices[:calib_dset_size]

        if test_dset_size is not None:
            assert (calib_dset_size + test_dset_size <= len(indices),
                    f"size of calibration ({calib_dset_size}) + test dataset ({test_dset_size}) sizes "
                    f"exceed given dataset size.")

            test_indices = indices[calib_dset_size: calib_dset_size + test_dset_size]
        else:
            test_indices = indices[calib_dset_size:]

        self.calib_dataset = Dataset.from_pandas(self.dataset.iloc[calib_indices.tolist()])
        self.test_dataset = Dataset.from_pandas(self.dataset.iloc[test_indices.tolist()])

        # Format the datasets
        cf = CoTFormat.from_model_name(self.llm_bundle.llm_name)
        self.ff_list = [self.__suc_response_formats,
                        self.__uc_response_formats,
                        self.__nt_response_formats]
        self.response_fmt, self.numeric_conf_fmt, self.worded_conf_fmt = self.ff_list[cf.value](
            COT_SYSTEM_PROMPT,
            WORDED_CONF_PROMPT,
            NUMERIC_CONF_PROMPT,
            QUESTION_FORMAT,
            FINAL_ANSWER_FORMAT
        )

        self.calib_dataset = self.calib_dataset.map(self.response_fmt, batched=True)
        self.test_dataset = self.test_dataset.map(self.response_fmt, batched=True)

    def get_name(self):
        return self.__class__.__name__

    def get_calibration_and_test_data(self, batch_size=1, recompute=False):
        """
        Gets the logits and tokens from the llm over the calibration and test datasets.
        No EOS tokens are filtered at all.
        :param batch_size: generation batch size for both calib and test sets.
        :param recompute: whether to recompute the logits for both sets.
        :return:
        """
        print("Getting Calibration and Test data.")
        calib_filepath = self.target_dir / "calibration_data.pt"
        test_filepath = self.target_dir / "test_data.pt"

        if calib_filepath.exists() and not recompute:
            print(f"Found existing calibration data in {calib_filepath}")
            calib_conf_dset = torch.load(calib_filepath)
        else:
            print(f"Calibration data at ({calib_filepath}) not found.")
            self.llm_bundle.load_model()
            calib_logits_tokens = self.llm_bundle.get_tokens_and_logits_from_dset(self.calib_dataset,
                                                                                  batch_size=batch_size,
                                                                                  desc="Get Logits + Tokens (Calib)")
            calib_logits_tokens["answer"] = torch.Tensor(self.calib_dataset["answer"])

            calib_verbalised_dset = (self.calib_dataset
                                     .map(self.numeric_conf_fmt, batched=True)
                                     .map(self.worded_conf_fmt, batched=True))
            calib_verbalised_confs = self.llm_bundle.get_verbalised_confs_from_dset(calib_verbalised_dset,
                                                                                    batch_size=batch_size,
                                                                                    desc="Get Verbalised Confs (Calib)")

            # Obtain answers and logits confidences.
            calib_logit_confs_answers = self.llm_bundle.get_logits_confs_and_answers_from_dset(calib_logits_tokens)

            calib_conf_dset = self.calib_dataset.remove_columns(["question", "response_formatted"]).to_dict()
            calib_conf_dset.update(calib_logits_tokens)
            calib_conf_dset.update(calib_verbalised_confs)
            calib_conf_dset.update(calib_logit_confs_answers)

            torch.save(calib_conf_dset, str(self.target_dir / "calibration_data.pt"))
            print("calibration data done.")

        if test_filepath.exists() and not recompute:
            print(f"Found existing test data in {test_filepath}")
            test_conf_dset = torch.load(test_filepath)
        else:
            print(f"test data at ({test_filepath}) not found.")
            self.llm_bundle.load_model()
            test_logits_tokens = self.llm_bundle.get_tokens_and_logits_from_dset(self.test_dataset,
                                                                                  batch_size=batch_size,
                                                                                  desc="Get Logits + Tokens (Test)")
            test_logits_tokens["answer"] = torch.Tensor(self.test_dataset["answer"])

            test_verbalised_dset = (self.test_dataset
                                    .map(self.numeric_conf_fmt, batched=True)
                                    .map(self.worded_conf_fmt, batched=True))
            test_verbalised_confs = self.llm_bundle.get_verbalised_confs_from_dset(test_verbalised_dset,
                                                                                   batch_size=batch_size,
                                                                                   desc="Get Verbalised Confs (Test)")

            # Obtain answers and logits confidences.
            test_logit_confs_answers = self.llm_bundle.get_logits_confs_and_answers_from_dset(test_logits_tokens)

            test_conf_dset = self.test_dataset.remove_columns(["question", "response_formatted"]).to_dict()
            test_conf_dset.update(test_logits_tokens)
            test_conf_dset.update(test_verbalised_confs)
            test_conf_dset.update(test_logit_confs_answers)

            torch.save(test_conf_dset, str(self.target_dir / "test_data.pt"))
            print("test data done.")

        return DictDataset(calib_conf_dset), DictDataset(test_conf_dset)

    def run_calibration_pipeline(self,
                                 calibrator_type: Type[Calibrator],
                                 batch_size=1,
                                 recompute_logits=False,
                                 recalibrate=False) -> Tuple[DictDataset, DictDataset]:
        # Try to get logits and tokens for both calib and test
        calib_data, test_data = self.get_calibration_and_test_data(batch_size,
                                                                   recompute=recompute_logits)

        self.__calibrator = calibrator_type(self.llm_bundle)

        # Perhaps check for weights in the calibrator itself?
        # Some calibrators have no weights.
        weights_path = self.target_dir / self.__calibrator.get_name()
        print(f"weights path is {weights_path}")
        if (weights_path / "calib_weights.pt").exists() and not recalibrate:
            self.__calibrator.load(str(weights_path / "calib_weights.pt"))
        else:
            print("Performing calibration of model.")

            weights_path.mkdir(parents=True, exist_ok=True)
            self.__calibrator.calibrate(calibration_dset=calib_data,
                                        batch_size=batch_size)
            self.__calibrator.save(str(weights_path / "calib_weights.pt"))

        # test the calibrator.
        if (weights_path / "calib_results.pt").exists():
            print(f"Found existing calibration results at {weights_path / 'calib_results.pt'}")
            calib_confs = torch.load(weights_path / "calib_results.pt")
        else:
            print("Testing Calibrator on Calibration Dataset")
            calib_confs = self.__calibrator.test(test_dset=calib_data,
                                                 batch_size=batch_size)
            torch.save(calib_confs, weights_path / "calib_results.pt")

        if (weights_path / "test_results.pt").exists():
            print(f"Found existing test results at {weights_path / 'test_results.pt'}")
            test_confs = torch.load(weights_path / "test_results.pt")
        else:
            print("Testing Calibrator on Test Dataset")
            test_confs = self.__calibrator.test(test_dset=test_data,
                                                batch_size=batch_size)
            torch.save(test_confs, weights_path / "test_results.pt")

        calib_data.data_dict["calibrated_confs"] = calib_confs
        test_data.data_dict["calibrated_confs"] = test_confs

        return calib_data, test_data

    def get_calibrator_model(self):
        if self.__calibrator is None: return None
        return nn.Sequential(self.llm_bundle.llm_model, self.__calibrator.calibrator_model)

    def get_results_path(self):
        if self.__calibrator is None: return None
        return self.target_dir / self.__calibrator.get_name()

    def __suc_response_formats(self,
                               system_prompt: str,
                               worded_conf_user_prompt: str,
                               numeric_conf_user_prompt: str,
                               question_prompt: str,
                               answer_format: str):
        def response_fmt(x):
            questions = x['question']
            formatted = []
            for question in questions:
                formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": question_prompt.format(question=question)}],
                    tokenize=False,
                    add_generation_prompt=True,
                    return_tensors="pt")
                formatted.append(formatted_q)
            return {"response_formatted": formatted}

        def choice_fmt(conf_user_prompt, feature_name):
            def verb_conf_fmt(x):
                questions = x['question']
                answers = x["answers"]
                formatted = []
                for question, answer in zip(questions, answers):
                    formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                        [{"role": "system", "content": f"{question_prompt.format(question=question)}\n"
                                                       f"{answer_format.format(answer=answer)}"},
                         {"role": "user", "content": conf_user_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                        return_tensors="pt")
                    formatted.append(formatted_q)
                return {feature_name: formatted}

            return verb_conf_fmt

        return (response_fmt,
                choice_fmt(numeric_conf_user_prompt, "numeric_conf_formatted"),
                choice_fmt(worded_conf_user_prompt, "worded_conf_formatted"))

    def __uc_response_formats(self,
                              system_prompt: str,
                              worded_conf_user_prompt: str,
                              numeric_conf_user_prompt: str,
                              question_prompt: str,
                              answer_format: str):
        def response_fmt(x):
            questions = x['question']
            formatted = []
            for question in questions:
                formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                    [{"role": "user",
                      "content": f"{system_prompt}\n\n"
                                 f"{question_prompt.format(question=question)}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                formatted.append(formatted_q)
            return {"response_formatted": formatted}

        def choice_fmt(conf_user_prompt, feature_name):
            def verb_conf_fmt(x):
                questions = x['question']
                answers = x["answer"]
                formatted = []
                for question, answer in zip(questions, answers):
                    formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                        [{"role": "user",
                          "content": f"{question_prompt.format(question=question)}\n"
                                     f"{answer_format.format(answer=answer)}\n\n"
                                     f"{conf_user_prompt}"}],
                        tokenize=False,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )
                    formatted.append(formatted_q)
                return {feature_name: formatted}

            return verb_conf_fmt

        return (response_fmt,
                choice_fmt(numeric_conf_user_prompt, "numeric_conf_formatted"),
                choice_fmt(worded_conf_user_prompt, "worded_conf_formatted"))

    def __nt_response_formats(self,
                              system_prompt: str,
                              worded_conf_user_prompt: str,
                              numeric_conf_user_prompt: str,
                              question_prompt: str,
                              answer_format: str):
        def response_fmt(x):
            questions = x['question']
            formatted = []
            for question in questions:
                formatted_q = f"{system_prompt}\n\n{question_prompt.format(question=question)}"
                formatted.append(formatted_q)
            return {"response_formatted": formatted}

        def choice_fmt(conf_user_prompt, feature_name):
            def verb_conf_fmt(x):
                questions = x['question']
                answers = x["answer"]
                formatted = []
                for question, answer in zip(questions, answers):
                    formatted_q = (f"{question_prompt.format(question=question)}\n"
                                   f"{answer_format.format(answer=answer)}\n\n"
                                   f"{conf_user_prompt}")
                    formatted.append(formatted_q)
                return {feature_name: formatted}

            return verb_conf_fmt

        return (response_fmt,
                choice_fmt(numeric_conf_user_prompt, "numeric_conf_formatted"),
                choice_fmt(worded_conf_user_prompt, "worded_conf_formatted"))


classes = inspect.getmembers(sys.modules[__name__], class_predicate(InputFormatter))
input_formatter_dict: dict[str, InputFormatter.__class__] = {x: y for x, y in classes}

if __name__ == "__main__":
    x = GSMCoT(TextGenLLMBundle("google/gemma-1.1-2b-it"), 100, 100)
    print(x.calib_dataset[0].keys())