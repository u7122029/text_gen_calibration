from os import PathLike
from typing import Iterable

import dill

from data.gsm import get_gsm
from torch.utils.data import DataLoader, Dataset
import torch


def get_dataset(name):
    out = None
    if name == "GSM":
        out = get_gsm()
    return out


class DictDataset(Dataset):
    def __init__(self, data_dict: dict):
        assert len(data_dict.keys()) > 0
        self.ref_key = list(data_dict.keys())[0]
        for key in data_dict.keys():
            assert len(data_dict[key]) == len(data_dict[self.ref_key]), \
                f"ref key {self.ref_key} has {len(data_dict[self.ref_key])}, but key {key} has {len(data_dict[key])}."

        self.data_dict = data_dict
        self.get_keys = self.data_dict.keys()

    @classmethod
    def from_file(cls, path: PathLike):
        with open(path, "rb") as f:
            d = dill.load(f)
        return cls(d)

    def add_column(self, column_name: str, data):
        assert isinstance(column_name, str)
        assert len(data) == len(self.data_dict[self.ref_key])
        self.data_dict[column_name] = data

    def __len__(self):
        return len(self.data_dict[self.ref_key])

    def __getitem__(self, idx):
        return {k: self.data_dict[k][idx] for k in self.get_keys}

    def __str__(self):
        keys_str = f"keys: {self.data_dict.keys()}"
        types_dict = {}
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                types_dict[k] = (type(v), v.shape)
            else:
                types_dict[k] = type(v)
        return (f"Dataset(\n"
                f"\t{keys_str}\n"
                f"\t{types_dict}\n"
                f")")

    def __contains__(self, item):
        return item in self.data_dict

    def keys(self):
        return self.data_dict.keys()

    def collate_fn(self, *keys, postprocess_fn=None):
        """
        Generates a collate function based on the keys provided.
        Also allows postprocessing.
        @param keys: The keys to include in the batch dictionary.
        @param postprocess_fn: The postprocessing function. Should take in the batched dictionary and output a processed version.
        @return: The batched dictionary.
        """
        def fn(data_list):
            out_dict = {k: [] for k in keys}
            for d in data_list:
                for k in keys:
                    out_dict[k].append(d[k])
            if postprocess_fn is not None:
                out_dict = postprocess_fn(out_dict)
            return out_dict
        return fn


if __name__ == "__main__":
    import torch
    x = DictDataset({"a": [1,2,3,4,5,6], "b": torch.rand(6)})

    dl = DataLoader(x, batch_size=4)
    print(next(iter(dl)))