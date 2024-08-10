import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from data import DictDataset
from utils import QUALITATIVE_SCALE, HF_TOKEN, DEVICE


class VerbalisedConfidence(Enum):
    NUMERIC = 0
    WORDED = 1


def extract_verbalized_confidence(expressions: List[str],
                                  mode: VerbalisedConfidence,
                                  expression_mapping: Optional[Dict[str, float]] = None
                                  ) -> Tuple[List[float], List[bool]]:
    """
    Extract the confidence scores from the verbalized confidence generated from a model.
    Taken from https://github.com/parameterlab/apricot/blob/6aea8510ca0da04a27abbb4b5ba39f727c26c342/src/eval.py#L270

    Parameters
    ----------
    expressions: List[str]
        List of expressions containing verbalized confidence.
    mode: str
        Whether the confidence is "qualitative" or "quantitative". Defaults to the latter.
    expression_mapping: Optional[Dict[str, float]]
        If the mode is "qualitative", supply a dictionary that maps from confidence expression to numerical values.

    Returns
    -------
    Tuple[List[float], List[bool]]
        Extracted confidence scores, as well as list of boolean values indicating whether the extraction was successful.
    """
    assert isinstance(mode, VerbalisedConfidence), f"Mode has to be a VerbalisedConfidence, but {mode} found."
    if expression_mapping is None:
        expression_mapping = QUALITATIVE_SCALE

    if mode == VerbalisedConfidence.WORDED:
        assert (expression_mapping is not None), "'expression_mapping' has to be specified for qualitative mode."

    confidences, successful = [], []

    for expression in expressions:
        if mode == VerbalisedConfidence.WORDED:
            template = rf"({'|'.join(expression_mapping.keys())})"
        else:
            # With the template below, try to capture anything like: 95%, 95 %, 96.666, 100, etc.
            template = r"\d{1,3}(?:\.\d+)?\s?\%?"

        try:
            res = re.search(template, expression).group(0)

            if mode == VerbalisedConfidence.WORDED:
                conf = expression_mapping[res]
            else:
                conf = float(res.replace("%", "")) / 100
                if not (0 <= conf <= 1):
                    successful.append(False)
                    confidences.append(-1)
                    continue

            successful.append(True)
            confidences.append(conf)
        except AttributeError:
            successful.append(False)
            confidences.append(-1)
    assert len(expressions) == len(confidences), f"length of expressions not equal to that of the outputted confidences ({len(expressions)} vs. {len(confidences)})"
    assert len(successful) == len(confidences)
    return confidences, successful


class LLMBundle(ABC):
    def __init__(self, llm_name: str):
        self.llm_name = llm_name

        self.tokeniser = AutoTokenizer.from_pretrained(self.llm_name,
                                                       token=HF_TOKEN,
                                                       padding_side="left")
        self.llm_model = None
        self.tokeniser.pad_token_id = self.tokeniser.eos_token_id

    def vocab_size(self):
        manual_sizes = {
            "microsoft/Phi-3-mini-4k-instruct": 32064
        }
        if self.llm_name in manual_sizes:
            return manual_sizes[self.llm_name]
        return len(self.tokeniser)

    @abstractmethod
    def get_model(self):
        pass

    def load_model(self):
        """
        Calls the function to load the model into the program. This is a whole separate method because a user might only
        need the tokeniser.
        :return:
        """
        if self.llm_model is None:
            print(f"Loading model {self.llm_name}")
            self.get_model()
        else:
            print(f"Model {self.llm_name} already loaded.")

    def __del__(self):
        # free up memory.
        del self.tokeniser, self.llm_model


class TextGenLLMBundle(LLMBundle):
    def get_model(self):
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_name,
                                                              device_map="auto",
                                                              torch_dtype=torch.float16,
                                                              token=HF_TOKEN)

    def get_tokens_and_logits_from_dset(self, dset: DictDataset, batch_size=1, max_new_tokens=550, desc=None):
        """
        Generate the

        - Responses,
        - Logits,
        - Verbalised numerical/quantitative confidences,
        - Verbalised worded/qualitative confidences.
        Over a given dataloader.
        :param dset:
        :param batch_size:
        :param max_new_tokens:
        :param desc:
        :return:
        """
        print("Getting logits and tokens.")
        all_response_logits = []
        all_response_tokens = []
        dl = DataLoader(dset, batch_size=batch_size)

        # Logits and Output Tokens
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=desc):
            formatted = batch["response_formatted"]

            inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
            generated = self.llm_model.generate(**inputs,
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

                all_response_logits.append(processed_logits)
                all_response_tokens.append(processed_response)
        out_dict = {}
        out_dict.update({"logits": all_response_logits,
                         "tokens": all_response_tokens})

        return out_dict

    def get_verbalised_confs_from_dset(self, dset: DictDataset, batch_size=1, max_new_tokens=30, desc=None):
        """

        :param dset:
        :param batch_size:
        :param max_new_tokens:
        :param desc:
        :return:
        """
        out_dict = {
            "numeric_confs": [],
            "numeric_successful": [],
            "worded_confs": [],
            "worded_successful": []
        }

        dl = DataLoader(dset, batch_size=batch_size)

        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=desc):
            numeric_formatted = batch["numeric_conf_formatted"]
            worded_formatted = batch["worded_conf_formatted"]

            inputs_numeric = self.tokeniser(numeric_formatted, return_tensors="pt", padding=True).to(DEVICE)
            inputs_worded = self.tokeniser(worded_formatted, return_tensors="pt", padding=True).to(DEVICE)
            numeric_generated = self.llm_model.generate(**inputs_numeric,
                                                        max_new_tokens=max_new_tokens,
                                                        return_dict_in_generate=True,
                                                        pad_token_id=self.tokeniser.eos_token_id)
            worded_generated = self.llm_model.generate(**inputs_worded,
                                                       max_new_tokens=max_new_tokens,
                                                       return_dict_in_generate=True,
                                                       pad_token_id=self.tokeniser.eos_token_id)

            # get the tokens, then remove the ones that made up the input.
            numeric_sequences = numeric_generated.sequences.cpu()
            worded_sequences = worded_generated.sequences.cpu()

            numeric_responses = self.tokeniser.batch_decode(
                numeric_sequences[:, inputs_numeric.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            worded_responses = self.tokeniser.batch_decode(
                worded_sequences[:, inputs_worded.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            n_confidences, n_successful = extract_verbalized_confidence(numeric_responses,
                                                                        VerbalisedConfidence.NUMERIC)
            w_confidences, w_successful = extract_verbalized_confidence(worded_responses,
                                                                        VerbalisedConfidence.WORDED)

            out_dict["numeric_confs"].extend(n_confidences)
            out_dict["numeric_successful"].extend(n_successful)
            out_dict["worded_confs"].extend(w_confidences)
            out_dict["worded_successful"].extend(w_successful)

        out_dict = {k: torch.Tensor(v) for k, v in out_dict.items()}
        out_dict["numeric_successful"] = out_dict["numeric_successful"].bool()
        out_dict["worded_successful"] = out_dict["worded_successful"].bool()
        return out_dict

    def get_logits_confs_and_answers_from_dset(self, logits_and_tokens: dict, correctness_func):
        """

        :param logits_and_tokens:
        :return:
        """
        all_preds = []
        all_successful = []
        all_confs = []
        for logits, tokens in zip(logits_and_tokens["logits"], logits_and_tokens["tokens"]):
            prob_vecs = torch.softmax(logits, dim=1)  # response_idx, response length, vocab_size
            tokens = tokens.cpu()
            decoded_response = self.tokeniser.decode(tokens)

            token_confidences = torch.take_along_dim(prob_vecs,
                                                     tokens.unsqueeze(1),
                                                     dim=1).squeeze(1)
            response_confidence = torch.mean(token_confidences).item()

            # TODO: Perhaps make getting the model's answer part of the input formatter, and leave
            #       the logit confidence to this class.
            decoded_response = decoded_response.lower()
            try:
                s1 = decoded_response.split("**explanation:**")[1]
                explanation, final_answer_raw = s1.split("**final answer:**")
                final_answer = re.findall(r"\d+", final_answer_raw)[0]
                successful = True
            except:
                final_answer = "-1" # Indicates a failed response.
                successful = False

            all_preds.append(final_answer)
            all_successful.append(successful)
            all_confs.append(response_confidence)

        # require logits_and_tokens["correct"] to be a torch tensor of type torch.uint8, each element in {0, 1}.
        logits_and_tokens["correct"] = correctness_func(all_preds, logits_and_tokens["answer"])
        logits_and_tokens["logits_confs"] = torch.Tensor(all_confs)
        logits_and_tokens["pred_successful"] = torch.Tensor(all_successful).bool()
        return logits_and_tokens