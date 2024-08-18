import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Tuple

from transformers import AutoTokenizer

from utils import QUALITATIVE_SCALE, HF_TOKEN


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
    assert (len(expressions) == len(confidences),
            f"length of expressions not equal to that of the outputted confidences "
            f"({len(expressions)} vs. {len(confidences)})")
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

