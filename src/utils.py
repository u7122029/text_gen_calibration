from typing import List, Optional, Dict, Tuple

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import inspect
from datasets import Dataset
import re
from enum import Enum

QUALITATIVE_SCALE = {
    "Very low": 0,
    "Low": 0.3,
    "Somewhat low": 0.45,
    "Medium": 0.5,
    "Somewhat high": 0.65,
    "High": 0.7,
    "Very high": 1,
}

RESULTS_PATH = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUMERIC_CONF_PROMPT = "Provide your confidence in the above answer only as a percentage (0-100%).\n**Confidence:**"
WORDED_CONF_PROMPT = (f"Provide your confidence in the above answer only as one of "
                      f"{' / '.join([f'{exp}' for exp in QUALITATIVE_SCALE.keys()])}.\n**Confidence:**")
COT_SYSTEM_PROMPT = ("You are a friendly chatbot that only outputs in the form:\n"
                     "**Explanation:** <Your explanation>\n"
                     "**Final Answer:** <A single number>")
FINAL_ANSWER_FORMAT = "**Final Answer:** {answer}"
QUESTION_FORMAT = "**Question:** {question}"


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
                    continue

            successful.append(True)
            confidences.append(conf)
        except AttributeError:
            successful.append(False)

    return confidences, successful


class TextGenLLMBundle:
    def __init__(self, llm_name: str):
        self.llm_name = llm_name

        # Get token.
        with open("token.txt") as f:
            self.token = f.read().strip()

        self.tokeniser = AutoTokenizer.from_pretrained(self.llm_name,
                                                       token=self.token,
                                                       padding_side="left")
        self.tokeniser.pad_token_id = self.tokeniser.eos_token_id
        self.llm_model = None

    def load_model(self):
        """
        Calls the function to load the model into the program. This is a whole separate method because a user might only
        need the tokeniser.
        :return:
        """
        print(f"Loading model {self.llm_name}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_name,
                                                              device_map="auto",
                                                              torch_dtype=torch.float16,
                                                              token=self.token)

    def is_model_loaded(self):
        return self.llm_model is not None

    def vocab_size(self):
        manual_sizes = {
            "microsoft/Phi-3-mini-4k-instruct": 32064
        }
        if self.llm_name in manual_sizes:
            return manual_sizes[self.llm_name]
        return len(self.tokeniser)

    def get_tokens_and_logits_from_dl(self, dset, batch_size=1, max_new_tokens=550, desc=None):
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

        #all_responses_decoded = self.tokeniser.batch_decode(all_response_tokens)
        #numeric_conf_prompts = [f"{decoded_response}" for decoded_response in zip(all_responses_decoded)]

        out_dict = {
            "response_logits": all_response_logits,
            "response_tokens": all_response_tokens
        }
        out_dset = Dataset.from_dict(out_dict)
        out_dset.add_column("question", dset["question"])

        return out_dset

    def get_verbalised_confs_from_dl(self, dset: Dataset, batch_size=1, max_new_tokens=30, desc=None):
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

        return Dataset.from_dict(out_dict)

    def __del__(self):
        # free up memory.
        del self.tokeniser, self.llm_model


class AbsModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class TokenLogitsDataset(torch.utils.data.Dataset):
    def __init__(self, logits, tokens, correct):
        """

        :param logits: List of tensors. The length of the list is the number of responses, and the shape of each tensor
        is [response_length (num_tokens), vocab_size]
        :param tokens: List of tensors. The length of the list is the number of responses, and the shape of each tensor
        is [response_length (num_tokens)]
        :param correct: Tensor involving boolean values of shape [num_responses].
        """
        self.logits = logits
        self.tokens = tokens
        self.correct = correct

        self.vocab_size = self.logits[0].shape[1]

        assert len(self.logits) == len(self.tokens), \
            f"given logits is not the same length as the tokens. len(logits): {len(self.logits)}, len(tokens): {len(self.tokens)}"
        assert len(self.tokens) == len(self.correct), \
            f"given tokens is not the same length as the labels. len(tokens): {len(self.tokens)}, len(correct): {len(self.correct)}."

        self.correct_vectors = []
        for t, c in zip(self.tokens, self.correct):
            vec = torch.zeros(len(t)) + c
            self.correct_vectors.append(vec)

    def __getitem__(self, item):
        return self.logits[item], self.tokens[item], self.correct_vectors[item]

    def __len__(self):
        return len(self.correct)

    @staticmethod
    def collate_fn(data):
        logits = []
        tokens = []
        correct_vecs = []
        for x in data:
            logits.append(x[0])
            tokens.append(x[1])
            correct_vecs.append(x[2])
        return torch.cat(logits), torch.cat(tokens), torch.cat(correct_vecs)


class TLTokenFrequencyDataset(TokenLogitsDataset):
    def __getitem__(self, item):
        tokens = self.tokens[item]
        relative_tfs = torch.zeros(self.vocab_size)
        token_counts = torch.bincount(tokens)
        relative_tfs[:len(token_counts)] += token_counts
        relative_tfs /= torch.sum(relative_tfs)
        return self.logits[item], tokens, self.correct_vectors[item], relative_tfs

    @staticmethod
    def collate_fn(data):
        logits = []
        tokens = []
        correct_vecs = []
        relative_tfs = []
        for x in data:
            logits.append(x[0])
            tokens.append(x[1])
            correct_vecs.append(x[2])
            relative_tfs.append(x[3])
        return logits, tokens, torch.cat(correct_vecs), torch.stack(relative_tfs)


def get_class_bases(x):
    bases = set()
    for base in x.__bases__:
        bases.add(base)
        bases = bases.union(get_class_bases(base))
    return bases


def class_predicate(cls):
    def predicate_func(x):
        if not inspect.isclass(x): return False

        class_bases = get_class_bases(x)
        return cls in class_bases

    return predicate_func
