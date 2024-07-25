from os import PathLike
from typing import List, Optional, Dict, Tuple, Any
import dill

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import inspect
import datasets
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
        if self.llm_model is None:
            print(f"Loading model {self.llm_name}")
            self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_name,
                                                                  device_map="auto",
                                                                  torch_dtype=torch.float16,
                                                                  token=self.token)
        else:
            print(f"Model {self.llm_name} already loaded.")

    def vocab_size(self):
        manual_sizes = {
            "microsoft/Phi-3-mini-4k-instruct": 32064
        }
        if self.llm_name in manual_sizes:
            return manual_sizes[self.llm_name]
        return len(self.tokeniser)

    def get_tokens_and_logits_from_dset(self, dset: datasets.Dataset, batch_size=1, max_new_tokens=550, desc=None):
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

    def get_verbalised_confs_from_dset(self, dset: datasets.Dataset, batch_size=1, max_new_tokens=30, desc=None):
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
        return out_dict

    def get_logits_confs_and_answers_from_dset(self, logits_and_tokens: dict):
        """

        :param logits_and_tokens:
        :return:
        """
        all_preds = []
        all_confs = []
        for logits, tokens in zip(logits_and_tokens["logits"], logits_and_tokens["tokens"]):
            prob_vecs = torch.softmax(logits, dim=1)  # response_idx, response length, vocab_size
            tokens = tokens.cpu()
            decoded_response = self.tokeniser.decode(tokens)

            token_confidences = torch.take_along_dim(prob_vecs,
                                                     tokens.unsqueeze(1), dim=1).squeeze(1)
            response_confidence = torch.mean(token_confidences).item()

            # TODO: Perhaps make getting the model's answer part of the input formatter, and leave
            #       the logit confidence to this class.
            decoded_response = decoded_response.lower()
            try:
                s1 = decoded_response.split("**explanation:**")[1]
                explanation, final_answer_raw = s1.split("**final answer:**")
                final_answer = int(re.findall(r"\d+", final_answer_raw)[0])
            except:
                final_answer = -1 # Indicates a failed response.

            all_preds.append(final_answer)
            all_confs.append(response_confidence)

        logits_and_tokens["correct"] = (torch.Tensor(all_preds) == logits_and_tokens["answer"]).to(torch.uint8)
        logits_and_tokens["logits_confs"] = torch.Tensor(all_confs)
        return logits_and_tokens

    def __del__(self):
        # free up memory.
        del self.tokeniser, self.llm_model


class AbsModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)


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


def dill_load(pth: PathLike) -> Any:
    with open(pth, "rb") as f:
        out = dill.load(f)
    return out


def dill_save(obj: Any, pth: PathLike):
    with open(pth, "wb") as f:
        dill.dump(obj, f)