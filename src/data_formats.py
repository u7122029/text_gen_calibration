from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re


def get_gsm():
    raw_dataset = load_dataset("gsm8k", "main", split="test").to_pandas()
    raw_dataset["answer"] = raw_dataset["answer"].apply(lambda x: int(re.sub(r'[^\w\s]', '', x.split("####")[1])))

    return raw_dataset


def get_dataset(name):
    out = None
    if name == "GSM":
        out = get_gsm()
    return out


if __name__ == "__main__":
    get_dataset("GSM")