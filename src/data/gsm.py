import re

import torch
from datasets import load_dataset

from .dictdataset import DictDataset


def get_gsm():
    raw_dataset = load_dataset("gsm8k", "main", split="test").to_pandas()
    raw_dataset["answer"] = raw_dataset["answer"].apply(lambda x: int(re.sub(r'[^\w\s]', '', x.split("####")[1])))
    raw_dataset["id"] = torch.Tensor(range(len(raw_dataset)))

    return DictDataset(raw_dataset.to_dict())