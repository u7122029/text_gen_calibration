import torch
from datasets import load_dataset
from .dictdataset import DictDataset


def get_answer(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    right_brace_idx = None
    num_left_braces_open = 0
    left_brace_idx = None
    for i in range(idx, len(string)):
        if string[i] == "{":
            if num_left_braces_open == 0 and left_brace_idx is None:
                left_brace_idx = i
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[left_brace_idx + 1:right_brace_idx]

    return retval


def get_math():
    dataset = DictDataset(load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True).to_dict())
    dataset["question"] = dataset["problem"]
    del dataset["problem"]
    del dataset["type"]
    dataset["answer"] = [get_answer(x) for x in dataset["solution"]]
    del dataset["solution"]
    del dataset["level"]
    dataset["id"] = torch.Tensor(range(len(dataset)))

    return dataset
