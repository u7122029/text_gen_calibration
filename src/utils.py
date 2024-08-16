from os import PathLike
from pathlib import Path
from typing import Any
import dill

import torch
import inspect
import warnings

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

try:
    with open("hf_token.txt") as f:
        HF_TOKEN = f.read().strip()
except:
    warnings.warn(f"Huggingface token from file hf_token.txt not found. Some models requiring such a token will not be"
                  f"loaded.")


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


def dill_save(obj: Any, pth: Path):
    pth.parent.mkdir(parents=True, exist_ok=True)
    with open(pth, "wb") as f:
        dill.dump(obj, f)