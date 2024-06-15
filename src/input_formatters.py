import torch
from torch import nn
import re
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from utils import generate_over_dataloader

RESULTS_PATH = Path("results")
LOGITS_SUBDIR = Path("logits")


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
            "mistralai/Mistral-7B-Instruct-v0.2": cls.USER_CHAT,
            "01-ai/Yi-1.5-9B-Chat": cls.SYSTEM_USER_CHAT,
            "NousResearch/Hermes-2-Theta-Llama-3-8B": cls.SYSTEM_USER_CHAT,
            "NousResearch/Hermes-2-Pro-Mistral-7B": cls.SYSTEM_USER_CHAT
        }
        return name_dict[name]


class GSMCoT:
    def __init__(self, model_name, dataset, token, calib_dset_size, test_dset_size=None):
        self.model_name = model_name
        self.dataset = dataset
        self.tokeniser = AutoTokenizer.from_pretrained(self.model_name, token=token, padding_side="left")
        self.tokeniser.pad_token_id = self.tokeniser.eos_token_id
        self.__calibrator_class = None

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
        cf = CoTFormat.from_model_name(model_name)
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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          device_map="auto",
                                                          torch_dtype=torch.float16,
                                                          token=token)
        self.__calibrator_model = None
        self.calibrator_type_used = None

    def get_name(self):
        return self.__class__.__name__

    def get_logits_and_tokens(self, batch_size=1, recompute=False):
        """
        Gets the logits and tokens from the model over the calibration and test datasets.
        No EOS tokens are filtered at all.
        :param batch_size: generation batch size for both calib and test sets.
        :param recompute: whether to recompute the logits for both sets.
        :return:
        """
        target_dir = RESULTS_PATH / LOGITS_SUBDIR
        target_dir.mkdir(parents=True, exist_ok=True)

        filepath = target_dir / f"{self.get_name()}.pt"
        if filepath.exists() and not recompute:
            d = torch.load(filepath)
            return d["all_logits_calib"], d["all_tokens_calib"], d["all_logits_test"], d["all_tokens_test"]

        dl_calib = DataLoader(self.calib_dataset, batch_size=batch_size)
        dl_test = DataLoader(self.test_dataset, batch_size=batch_size)

        calib_dict = generate_over_dataloader(dl_calib, self.tokeniser, self.model, desc="Get Logits + Tokens (Calib)")
        test_dict = generate_over_dataloader(dl_test, self.tokeniser, self.model, desc="Get Logits + Tokens (Test)")

        out_dict = {
            "all_logits_calib": calib_dict["all_logits"],
            "all_tokens_calib": calib_dict["test_tokens"],
            "all_logits_test": test_dict["all_logits"],
            "all_tokens_test": test_dict["test_tokens"]
        }
        torch.save(out_dict, str(RESULTS_PATH / LOGITS_SUBDIR / f"{self.get_name()}.pt"))
        return (out_dict["all_logits_calib"],
                out_dict["all_tokens_calib"],
                out_dict["all_logits_test"],
                out_dict["all_tokens_test"])

    def train_calibrator(self, calibrator_type, batch_size=1, recompute_logits=False):
        """
        Train the calibrator model on the calibration dataset.
        :param calibrator_type:
        :param batch_size:
        :param recompute_logits:
        :return:
        """
        self.calibrator_type_used = calibrator_type

        original_logits, original_tokens, _, _ = self.get_logits_and_tokens(batch_size, recompute=recompute_logits)

        all_tokens = []
        all_logits = []
        all_preds = []
        confs_before_calib = []
        for formatted, logits, tokens in zip(self.calib_dataset["formatted"], original_logits, original_tokens):
            out_dict = self.__process_generated_output(logits, tokens)

            all_logits.append(out_dict["processed_logits"])
            all_tokens.append(out_dict["processed_tokens"])
            all_preds.append(out_dict["final_answer"])
            confs_before_calib.append(out_dict["confidence"])

        confs_before_calibration = torch.Tensor(confs_before_calib)
        all_preds = torch.Tensor(all_preds)
        correct = all_preds == torch.Tensor(self.calib_dataset["answer"])
        self.__calibrator_class = calibrator_type(self.tokeniser, self.model, False)

        self.__calibrator_class.calibrate(
            all_tokens=all_tokens,
            all_logits=all_logits,
            correct=correct,
            batch_size=batch_size
        )
        confs_after_calibration = self.__calibrator_class.test(
            test_tokens=all_tokens,
            test_logits=all_logits,
            correct=correct,
        )

        return confs_before_calibration, confs_after_calibration, correct

    def test_calibrator(self, calibrator_type):
        """
        Test the calibrator model on the test dataset.
        :param calibrator_type:
        :param batch_size:
        :param recompute_logits:
        :return:
        """
        self.calibrator_type_used = calibrator_type

        _, _, original_logits, original_tokens = self.get_logits_and_tokens()
        all_tokens = []
        all_logits = []
        all_preds = []
        confs_before_calib = []
        for formatted, logits, tokens in zip(self.calib_dataset["formatted"], original_logits, original_tokens):
            out_dict = self.__process_generated_output(logits, tokens)

            all_logits.append(out_dict["processed_logits"])
            all_tokens.append(out_dict["processed_tokens"])
            all_preds.append(out_dict["final_answer"])
            confs_before_calib.append(out_dict["confidence"])

        confs_before_calibration = torch.Tensor(confs_before_calib)
        all_preds = torch.Tensor(all_preds)
        correct = all_preds == torch.Tensor(self.calib_dataset["answer"])
        #self.__calibrator = calibrator_type(self.tokeniser, self.model, False)

        confs_after_calibration = self.__calibrator_class.test(
            test_tokens=all_tokens,
            test_logits=all_logits,
            correct=correct,
        )

        return confs_before_calibration, confs_after_calibration, correct

    def get_calibrator_model(self):
        if self.__calibrator_model is None: return None
        return nn.Sequential(self.model, self.__calibrator_model)

    def __process_generated_output(self, logits, tokens):
        """
        Compute the model's answer,
        the probability vectors for each token outputted, and the eos mask used on the output
        :param logits: the generation logits for one prompt.
        :param tokens: the tokens for one prompt.
        :return:
        """
        prob_vecs = torch.softmax(logits, dim=1)  # response_idx, response length, vocab_size
        tokens = tokens.cpu()
        decoded_response = self.tokeniser.decode(tokens)

        eos_mask = tokens != self.tokeniser.eos_token_id

        processed_logits = logits[eos_mask]
        processed_response = tokens[eos_mask]
        prob_vecs_no_eos = prob_vecs[eos_mask]

        token_confidences = torch.take_along_dim(prob_vecs_no_eos,
                                                 processed_response.unsqueeze(1), dim=1).squeeze(1)
        response_confidence = torch.mean(token_confidences).item()

        decoded_response = decoded_response.lower()
        try:
            s1 = decoded_response.split("**explanation:**")[1]
            explanation, final_answer_raw = s1.split("**final answer:**")
            final_answer = int(re.findall(r"\d+", final_answer_raw)[0])
        except:
            final_answer = -1

        # Computing probabilities using the generated logits.
        out_dict = {
            #"explanations": explanations,
            "processed_tokens": processed_response,
            "processed_logits": processed_logits,
            "confidence": response_confidence,
            "final_answer": final_answer,
            "eos_mask": eos_mask
        }

        return out_dict

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
