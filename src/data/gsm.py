import re

from datasets import load_dataset


def get_gsm():
    raw_dataset = load_dataset("gsm8k", "main", split="test").to_pandas()
    raw_dataset["answer"] = raw_dataset["answer"].apply(lambda x: int(re.sub(r'[^\w\s]', '', x.split("####")[1])))
    raw_dataset["id"] = list(range(len(raw_dataset)))

    return raw_dataset