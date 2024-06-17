import torch
from torch import nn
import re
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from utils import RESULTS_PATH
from calibrators import Calibrator
from typing import Type, Optional


class CoTFormat(Enum):
    SYSTEM_USER_CHAT = 1
    USER_CHAT = 2
    NO_TEMPLATE = 3
    DOLLY_15K = 4

    @classmethod
    def from_model_name(cls, name):
        name_dict = {
            "google/gemma-1.1-2b-it": cls.USER_CHAT,
            "google/gemma-1.1-7b-it": cls.USER_CHAT,
            "HuggingFaceH4/zephyr-7b-beta": cls.SYSTEM_USER_CHAT,
            "meta-llama/Meta-Llama-3-8B-Instruct": cls.SYSTEM_USER_CHAT,
            "mistralai/Mistral-7B-Instruct-v0.3": cls.USER_CHAT,
            "01-ai/Yi-1.5-9B-Chat": cls.SYSTEM_USER_CHAT,
            "NousResearch/Hermes-2-Theta-Llama-3-8B": cls.SYSTEM_USER_CHAT,
            "NousResearch/Hermes-2-Pro-Mistral-7B": cls.SYSTEM_USER_CHAT
        }
        return name_dict[name]


class GSMCoT:
    def __init__(self, llm_name, dataset, token, calib_dset_size, test_dset_size=None):
        self.llm_name = llm_name
        self.token = token
        self.task_name = "GSM"
        self.dataset = dataset

        # tokeniser is needed for dataset transformation.
        self.tokeniser = AutoTokenizer.from_pretrained(self.llm_name, token=token, padding_side="left")
        self.tokeniser.pad_token_id = self.tokeniser.eos_token_id
        self.__calibrator: Optional[Calibrator] = None

        self.target_dir = Path(RESULTS_PATH) / self.llm_name / self.task_name
        self.target_dir.mkdir(parents=True, exist_ok=True)

        indices = torch.randperm(len(self.dataset))
        calib_indices = indices[:calib_dset_size]

        if test_dset_size is not None:
            assert calib_dset_size + test_dset_size <= len(indices), \
                f"size of calibration ({calib_dset_size}) + test dataset ({test_dset_size}) sizes exceed given dataset size."
            test_indices = indices[calib_dset_size: calib_dset_size + test_dset_size]
        else:
            test_indices = indices[calib_dset_size:]

        self.calib_dataset = Dataset.from_pandas(self.dataset.iloc[calib_indices.tolist()])
        self.test_dataset = Dataset.from_pandas(self.dataset.iloc[test_indices.tolist()])

        self.system_text = ("You are a friendly chatbot that only outputs in the form:\n"
                            "**Explanation:** <Your explanation>\n"
                            "**Final Answer:** <A single number>")

        # Format the datasets
        cf = CoTFormat.from_model_name(llm_name)
        if cf == CoTFormat.SYSTEM_USER_CHAT:
            format_func = self.__system_user_chat_format
        elif cf == CoTFormat.USER_CHAT:
            format_func = self.__user_chat_format
        elif cf == CoTFormat.NO_TEMPLATE:
            format_func = self.__no_template_format
        else:
            raise Exception(f"Invalid enum value {cf}")

        self.calib_dataset = self.calib_dataset.map(format_func, batched=True)
        self.test_dataset = self.test_dataset.map(format_func, batched=True)
        self.llm = None

    def load_llm(self):
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_name,
                                                        device_map="auto",
                                                        torch_dtype=torch.float16,
                                                        token=self.token)

    def get_name(self):
        return self.__class__.__name__

    def get_logits_and_tokens(self, batch_size=1, recompute=False):
        """
        Gets the logits and tokens from the llm over the calibration and test datasets.
        No EOS tokens are filtered at all.
        :param batch_size: generation batch size for both calib and test sets.
        :param recompute: whether to recompute the logits for both sets.
        :return:
        """

        filepath = self.target_dir / f"logits.pt"
        if filepath.exists() and not recompute:
            d = torch.load(filepath)
            return d["all_logits_calib"], d["all_tokens_calib"], d["all_logits_test"], d["all_tokens_test"]

        self.load_llm()
        dl_calib = DataLoader(self.calib_dataset, batch_size=batch_size)
        dl_test = DataLoader(self.test_dataset, batch_size=batch_size)

        calib_logits, calib_tokens = self.__generate_over_dataloader(dl_calib, desc="Get Logits + Tokens (Calib)")
        test_logits, test_tokens = self.__generate_over_dataloader(dl_test, desc="Get Logits + Tokens (Test)")

        out_dict = {
            "all_logits_calib": calib_logits,
            "all_tokens_calib": calib_tokens,
            "all_logits_test": test_logits,
            "all_tokens_test": test_tokens
        }
        torch.save(out_dict, str(filepath))
        return calib_logits, calib_tokens, test_logits, test_tokens

    def run_calibration_pipeline(self,
                                 calibrator_type: Type[Calibrator],
                                 batch_size=1,
                                 recompute_logits=False,
                                 recalibrate=False):
        # Try to get logits and tokens for both calib and test
        calib_logits, calib_tokens, test_logits, test_tokens = self.get_logits_and_tokens(batch_size,
                                                                                          recompute=recompute_logits)

        # Get answers and whether they are correct (calib).
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
        self.__calibrator = calibrator_type(self.tokeniser, self.llm, False)

        weights_path = self.target_dir / self.__calibrator.get_name()
        if (weights_path / "calib_weights.pt").exists() and not recalibrate:
            self.__calibrator.load(str(weights_path / "calib_weights.pt"))
        else:
            weights_path.mkdir(parents=True, exist_ok=True)
            self.__calibrator.calibrate(calib_tokens=calib_tokens,
                                        calib_logits=calib_logits,
                                        correct=calib_correct,
                                        batch_size=batch_size)
            self.__calibrator.save(str(weights_path / "calib_weights.pt"))

        # test the calibrator.
        calib_confs_after = self.__calibrator.test(test_tokens=calib_tokens,
                                                   test_logits=calib_logits,
                                                   correct=calib_correct,
                                                   batch_size=batch_size)

        test_confs_after = self.__calibrator.test(test_tokens=test_tokens,
                                                  test_logits=test_logits,
                                                  correct=test_correct,
                                                  batch_size=batch_size)

        return calib_confs_before, calib_confs_after, calib_correct, test_confs_before, test_confs_after, test_correct

    def get_calibrator_model(self):
        if self.__calibrator is None: return None
        return nn.Sequential(self.llm, self.__calibrator.calibrator_model)

    def get_results_path(self):
        if self.__calibrator is None: return None
        return self.target_dir / self.__calibrator.get_name()

    def __generate_over_dataloader(self, dl, max_new_tokens=550, desc=None):
        all_logits = []
        all_tokens = []
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=desc):
            formatted = batch["formatted"]

            inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
            generated = self.llm.generate(**inputs,
                                          max_new_tokens=max_new_tokens,
                                          output_logits=True,
                                          return_dict_in_generate=True,
                                          pad_token_id=self.tokeniser.eos_token_id)
            model_logits = torch.stack(generated.logits).permute(1, 0, 2).cpu()

            # get the tokens, then remove the ones that made up the input.
            sequences = generated.sequences.cpu()
            responses: torch.Tensor = sequences[:, inputs.input_ids.shape[1]:]

            for logits, response in zip(model_logits, responses):
                eos_mask = response != self.tokeniser.eos_token_id

                processed_logits = logits[eos_mask]
                processed_response = response[eos_mask]

                all_logits.append(processed_logits)
                all_tokens.append(processed_response)

        return all_logits, all_tokens

    def __process_generated_output(self, logits, tokens):
        """
        Compute the llm's answer and confidence.
        :param logits: the generation logits for one prompt.
        :param tokens: the tokens for one prompt.
        :return:
        """
        prob_vecs = torch.softmax(logits, dim=1)  # response_idx, response length, vocab_size
        tokens = tokens.cpu()
        decoded_response = self.tokeniser.decode(tokens)

        """eos_mask = tokens != self.tokeniser.eos_token_id

        processed_logits = logits[eos_mask]
        processed_response = tokens[eos_mask]
        prob_vecs_no_eos = prob_vecs[eos_mask]"""

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

    def __system_user_chat_format(self, x):
        questions = x['question']
        formatted = []
        for question in questions:
            formatted_q = self.tokeniser.apply_chat_template([{"role": "system", "content": self.system_text},
                                                              {"role": "user", "content": f"**Question:** {question}"}],
                                                             tokenize=False,
                                                             add_generation_prompt=True,
                                                             return_tensors="pt")
            formatted.append(formatted_q)
        return {"formatted": formatted}

    def __user_chat_format(self, x):
        questions = x['question']
        formatted = []
        for question in questions:
            formatted_q = self.tokeniser.apply_chat_template(
                [{"role": "user", "content": f"{self.system_text}\n\n**Question:** {question}"}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            formatted.append(formatted_q)
        return {"formatted": formatted}

    def __no_template_format(self, x):
        questions = x['question']
        formatted = []
        for question in questions:
            formatted_q = f"{self.system_text}\n\n**Question:** {question}\n"
            formatted.append(formatted_q)
        return {"formatted": formatted}
