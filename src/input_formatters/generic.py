import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Type, Tuple, Optional

import torch

from calibrators import Calibrator
from data import DictDataset
from utils import TextGenLLMBundle, dill_load, dill_save, RESULTS_PATH, COT_SYSTEM_PROMPT, WORDED_CONF_PROMPT, \
    NUMERIC_CONF_PROMPT, QUESTION_FORMAT, FINAL_ANSWER_FORMAT


class InputFormatter(ABC):
    """
    TODO: Determine methods that should be common across all subclasses.
    """

    @abstractmethod
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 dataset: DictDataset,
                 calib_dset_size: int,
                 test_dset_size: Optional[int] = None):
        """
        Abstract constructor to ensure that this class cannot be instantiated.
        """
        self.llm_bundle = llm_bundle
        self.dataset = dataset

        self.target_dir = Path(RESULTS_PATH) / self.llm_bundle.llm_name / self.__class__.__name__
        self.target_dir.mkdir(parents=True, exist_ok=True)

        self.__calibrator: Optional[Calibrator] = None

        indices = torch.randperm(len(self.dataset))
        calib_indices = indices[:calib_dset_size]

        if test_dset_size is not None:
            assert (calib_dset_size + test_dset_size <= len(indices),
                    f"size of calibration ({calib_dset_size}) + test dataset ({test_dset_size}) sizes "
                    f"exceed given dataset size.")

            test_indices = indices[calib_dset_size: calib_dset_size + test_dset_size]
        else:
            test_indices = indices[calib_dset_size:]

        self.calib_dataset: DictDataset = self.dataset[calib_indices]
        self.test_dataset: DictDataset = self.dataset[test_indices]

    @abstractmethod
    def get_calibration_and_test_data(self, batch_size, recompute=False):
        pass

    @abstractmethod
    def run_calibration_pipeline(self,
                                 calibrator_type: Type[Calibrator],
                                 batch_size=1,
                                 recompute_logits=False,
                                 recalibrate=False,
                                 **kwargs) -> Tuple[DictDataset, DictDataset]:
        pass


class CoTInputFormatter(InputFormatter, ABC):
    @abstractmethod
    def __init__(self, llm_bundle: TextGenLLMBundle, dataset: DictDataset, calib_dset_size, test_dset_size=None):
        """
        Abstract constructor to ensure that this class cannot be instantiated.
        """
        InputFormatter.__init__(self, llm_bundle, dataset, calib_dset_size, test_dset_size)

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

        self.calib_dataset.update(self.response_fmt(self.calib_dataset))
        self.test_dataset.update(self.response_fmt(self.test_dataset))

    def get_calibration_and_test_data(self, batch_size=1, recompute=False):
        """
        Gets the logits and tokens from the llm over the calibration and test datasets.
        No EOS tokens are filtered at all.
        :param batch_size: generation batch size for both calib and test sets.
        :param recompute: whether to recompute the logits for both sets.
        :return:
        """
        print("Getting Calibration and Test data.")
        calib_filepath = self.target_dir / "calibration_data.dill"
        test_filepath = self.target_dir / "test_data.dill"

        if calib_filepath.exists() and not recompute:
            print(f"Found existing calibration data in {calib_filepath}")
            calib_conf_dset = dill_load(calib_filepath)
            self.calib_dataset.update(calib_conf_dset)
        else:
            print(f"Calibration data at ({calib_filepath}) not found.")
            self.llm_bundle.load_model()
            with torch.no_grad():
                calib_logits_tokens = self.llm_bundle.get_tokens_and_logits_from_dset(self.calib_dataset,
                                                                                      batch_size=batch_size,
                                                                                      desc="Get Logits + Tokens (Calib)")
            calib_logits_tokens["answer"] = self.calib_dataset["answer"]

            (self.calib_dataset
             .update(self.numeric_conf_fmt(self.calib_dataset))
             .update(self.worded_conf_fmt(self.calib_dataset)))

            with torch.no_grad():
                calib_verbalised_confs = self.llm_bundle.get_verbalised_confs_from_dset(self.calib_dataset,
                                                                                        batch_size=batch_size,
                                                                                        desc="Get Verbalised Confs (Calib)")

            # Obtain answers and logits confidences.
            calib_logit_confs_answers = self.llm_bundle.get_logits_confs_and_answers_from_dset(calib_logits_tokens)

            self.calib_dataset.remove_columns(["response_formatted",
                                               "numeric_conf_formatted",
                                               "worded_conf_formatted"])
            self.calib_dataset.update(calib_logits_tokens)
            self.calib_dataset.update(calib_verbalised_confs)
            self.calib_dataset.update(calib_logit_confs_answers)

            dill_save(self.calib_dataset, self.target_dir / "calibration_data.dill")
            print("calibration data done.")

        if test_filepath.exists() and not recompute:
            print(f"Found existing test data in {test_filepath}")
            test_conf_dset = dill_load(test_filepath)
            self.test_dataset.update(test_conf_dset)
        else:
            print(f"test data at ({test_filepath}) not found.")
            self.llm_bundle.load_model()
            with torch.no_grad():
                test_logits_tokens = self.llm_bundle.get_tokens_and_logits_from_dset(self.test_dataset,
                                                                                     batch_size=batch_size,
                                                                                     desc="Get Logits + Tokens (Test)")
            test_logits_tokens["answer"] = torch.Tensor(self.test_dataset["answer"])

            (self.test_dataset
             .update(self.numeric_conf_fmt(self.test_dataset))
             .update(self.worded_conf_fmt(self.test_dataset)))
            with torch.no_grad():
                test_verbalised_confs = self.llm_bundle.get_verbalised_confs_from_dset(self.test_dataset,
                                                                                       batch_size=batch_size,
                                                                                       desc="Get Verbalised Confs (Test)")

            # Obtain answers and logits confidences.
            test_logit_confs_answers = self.llm_bundle.get_logits_confs_and_answers_from_dset(test_logits_tokens)

            self.test_dataset.remove_columns(["response_formatted",
                                              "numeric_conf_formatted",
                                              "worded_conf_formatted"])
            self.test_dataset.update(test_logits_tokens)
            self.test_dataset.update(test_verbalised_confs)
            self.test_dataset.update(test_logit_confs_answers)

            dill_save(self.test_dataset, self.target_dir / "test_data.dill")
            print("test data done.")

        return self.calib_dataset, self.test_dataset

    def run_calibration_pipeline(self,
                                 calibrator_type: Type[Calibrator],
                                 batch_size=1,
                                 recompute_logits=False,
                                 recalibrate=False,
                                 **kwargs) -> Tuple[DictDataset, DictDataset]:
        # Try to get logits and tokens for both calib and test
        calib_data, test_data = self.get_calibration_and_test_data(batch_size,
                                                                   recompute=recompute_logits)

        self.__calibrator = calibrator_type(self.llm_bundle)

        # Perhaps check for weights in the calibrator itself?
        # Some calibrators have no weights.
        weights_path = self.target_dir / self.__calibrator.get_name()
        cw_path = weights_path / "calib_weights.dill"
        if cw_path.exists() and not recalibrate:
            self.__calibrator.load(str(cw_path))
        else:
            print("Performing calibration of model.")

            weights_path.mkdir(parents=True, exist_ok=True)
            self.__calibrator.calibrate(calibration_dset=calib_data,
                                        batch_size=batch_size)
            self.__calibrator.save(str(cw_path))

        # test the calibrator.
        cr_path = weights_path / "calib_results.dill"
        if cr_path.exists():
            print(f"Found existing calibration results at {cr_path}")
            calib_results = dill_load(cr_path)
        else:
            print("Testing Calibrator on Calibration Dataset")
            calib_results = self.__calibrator.test(test_dset=calib_data,
                                                   batch_size=batch_size)
            dill_save(calib_results, cr_path)

        tr_path = weights_path / "test_results.dill"
        if tr_path.exists():
            print(f"Found existing test results at {tr_path}")
            test_results = dill_load(tr_path)
        else:
            print("Testing Calibrator on Test Dataset")
            test_results = self.__calibrator.test(test_dset=test_data,
                                                  batch_size=batch_size)
            dill_save(test_results, tr_path)

        calib_data.update(calib_results)
        test_data.update(test_results)

        return calib_data, test_data

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
