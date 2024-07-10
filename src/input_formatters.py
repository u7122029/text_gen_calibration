import torch
from torch import nn
import re
from datasets import Dataset
from enum import Enum
from torch.utils.data import DataLoader
from pathlib import Path

from data_formats import get_dataset
from utils import (RESULTS_PATH,
                   TextGenLLMBundle,
                   class_predicate,
                   COT_SYSTEM_PROMPT,
                   WORDED_CONF_PROMPT,
                   NUMERIC_CONF_PROMPT,
                   FINAL_ANSWER_FORMAT,
                   QUESTION_FORMAT)
from calibrators import Calibrator
from typing import Type, Optional
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
        calib_filepath = self.target_dir / f"calibration_data.hf"
        test_filepath = self.target_dir / f"test_data.hf"
        """if calib_filepath.exists() and test_filepath.exists() and not recompute:
            print(f"Found existing logits and tokens in {filepath}")
            d = torch.load(filepath)
            print("Successfully loaded logits.")
            return d["all_logits_calib"], d["all_tokens_calib"], d["all_logits_test"], d["all_tokens_test"]"""

        #print("No existing logits and tokens found. Will now generate them.")
        self.llm_bundle.load_model()

        if calib_filepath.exists() and not recompute:
            print(f"Found existing calibration data in {calib_filepath}")
            calib_conf_dset = Dataset.from_file(str(calib_filepath))
        else:
            print(f"Calibration data at ({calib_filepath}) not found. Generating data from calibration dataset.")
            calib_gen_dset = self.llm_bundle.get_tokens_and_logits_from_dl(self.calib_dataset,
                                                                           batch_size=batch_size,
                                                                           desc="Get Logits + Tokens (Calib)")
            calib_gen_dset = calib_gen_dset.map(self.numeric_conf_fmt, batched=True).map(self.worded_conf_fmt, batched=True)

            calib_conf_dset = self.llm_bundle.get_verbalised_confs_from_dl(calib_gen_dset,
                                                                           batch_size=batch_size,
                                                                           desc="Get Verbalised Confs (Calib)")
            calib_conf_dset.save_to_disk(str(self.target_dir / "calibration_data.hf"))

        if test_filepath.exists() and not recompute:
            print(f"Found existing test data in {test_filepath}")
            test_conf_dset = Dataset.from_file(str(test_filepath))
        else:
            print(f"Test data at ({test_filepath}) not found. Generating data from test dataset.")
            test_gen_dset = self.llm_bundle.get_tokens_and_logits_from_dl(self.test_dataset,
                                                                          batch_size=batch_size,
                                                                          desc="Get Logits + Tokens (Test)")
            test_gen_dset = test_gen_dset.map(self.numeric_conf_fmt, batched=True).map(self.worded_conf_fmt, batched=True)
            test_conf_dset = self.llm_bundle.get_verbalised_confs_from_dl(test_gen_dset,
                                                                          batch_size=batch_size,
                                                                          desc="Get Verbalised Confs (Calib)")
            test_conf_dset.save_to_disk(str(self.target_dir / "test_data.hf"))

        """out_dict = {
            "all_logits_calib": calib_logits,
            "all_tokens_calib": calib_tokens,
            "all_logits_test": test_logits,
            "all_tokens_test": test_tokens
        }
        torch.save(out_dict, str(filepath))"""

        return calib_conf_dset, test_conf_dset

    def run_calibration_pipeline(self,
                                 calibrator_type: Type[Calibrator],
                                 batch_size=1,
                                 recompute_logits=False,
                                 recalibrate=False):
        # Try to get logits and tokens for both calib and test
        calib_data, test_data = self.get_calibration_and_test_data(batch_size,
                                                                   recompute=recompute_logits)

        # Get answers and whether they are correct (calib).
        print("Getting answers from calibration set.")
        calib_preds = []
        calib_confs_before = []
        for formatted, logits, tokens in zip(self.calib_dataset["formatted"], calib_logits, calib_tokens):
            final_answer, confidence = self.__process_generated_output(logits, tokens)

            calib_preds.append(final_answer)
            calib_confs_before.append(confidence)
        calib_confs_before = torch.Tensor(calib_confs_before)
        calib_preds = torch.Tensor(calib_preds)
        calib_correct = calib_preds == torch.Tensor(self.calib_dataset["answer"])

        # Get answers and whether they are correct (test).
        print("Getting answers from test set.")
        test_preds = []
        test_confs_before = []
        for formatted, logits, tokens in zip(self.calib_dataset["formatted"], test_logits, test_tokens):
            final_answer, confidence = self.__process_generated_output(logits, tokens)

            test_preds.append(final_answer)
            test_confs_before.append(confidence)
        test_confs_before = torch.Tensor(test_confs_before)
        test_preds = torch.Tensor(test_preds)
        test_correct = test_preds == torch.Tensor(self.test_dataset["answer"])

        # perform calibration
        print("Initialising calibrator")
        self.__calibrator = calibrator_type(self.llm_bundle)

        weights_path = self.target_dir / self.__calibrator.get_name()
        if (weights_path / "calib_weights.pt").exists() and not recalibrate:
            self.__calibrator.load(str(weights_path / "calib_weights.pt"))
        else:
            print("Performing calibration of model.")
            weights_path.mkdir(parents=True, exist_ok=True)
            self.__calibrator.calibrate(calib_tokens=calib_tokens,
                                        calib_logits=calib_logits,
                                        correct=calib_correct,
                                        batch_size=batch_size)
            self.__calibrator.save(str(weights_path / "calib_weights.pt"))

        # test the calibrator.
        print("Testing Calibrator on Calibration Dataset")
        calib_confs_after = self.__calibrator.test(test_tokens=calib_tokens,
                                                   test_logits=calib_logits,
                                                   correct=calib_correct,
                                                   batch_size=batch_size)

        print("Testing Calibrator on Test Dataset")
        test_confs_after = self.__calibrator.test(test_tokens=test_tokens,
                                                  test_logits=test_logits,
                                                  correct=test_correct,
                                                  batch_size=batch_size)

        return calib_confs_before, calib_confs_after, calib_correct, test_confs_before, test_confs_after, test_correct

    def get_calibrator_model(self):
        if self.__calibrator is None: return None
        return nn.Sequential(self.llm_bundle.llm_model, self.__calibrator.calibrator_model)

    def get_results_path(self):
        if self.__calibrator is None: return None
        return self.target_dir / self.__calibrator.get_name()

    def __process_generated_output(self, logits, tokens):
        """
        Compute the llm's answer and confidence.
        :param logits: the generation logits for one prompt.
        :param tokens: the tokens for one prompt.
        :return:
        """
        prob_vecs = torch.softmax(logits, dim=1)  # response_idx, response length, vocab_size
        tokens = tokens.cpu()
        decoded_response = self.llm_bundle.tokeniser.decode(tokens)

        token_confidences = torch.take_along_dim(prob_vecs,
                                                 tokens.unsqueeze(1), dim=1).squeeze(1)
        response_confidence = torch.mean(token_confidences).item()

        decoded_response = decoded_response.lower()
        try:
            s1 = decoded_response.split("**explanation:**")[1]
            explanation, final_answer_raw = s1.split("**final answer:**")
            final_answer = int(re.findall(r"\d+", final_answer_raw)[0])
        except:
            final_answer = -1

        return final_answer, response_confidence

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