from os import PathLike
from pathlib import Path
from typing import Union, Iterable

import dill
import torch
from torch.utils.data import Dataset

from utils import dill_save


class DictDataset(Dataset):
    """
    Convert this to also extend from a dictionary.
    """
    def __init__(self, data_dict: dict):
        assert isinstance(data_dict, dict)
        assert len(data_dict.keys()) > 0

        self.ref_key = list(data_dict.keys())[0]
        for key in data_dict.keys():
            assert len(data_dict[key]) == len(data_dict[self.ref_key]), \
                f"ref key {self.ref_key} has {len(data_dict[self.ref_key])}, but key {key} has {len(data_dict[key])}."

        self.data_dict = data_dict

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls(kwargs)

    @classmethod
    def from_file(cls, path: PathLike):
        with open(path, "rb") as f:
            d = dill.load(f)
        return cls(d)

    def save(self, path: PathLike):
        dill_save(self.data_dict, path)

    def save_folderdset(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        digits_len = len(str(len(self)))
        for i, x in enumerate(self):
            p = path / f"{str(i).zfill(digits_len)}.dill"
            dill_save(x, p)

    def update(self, other: dict):
        for k, v in other.items():
            self[k] = v

        return self

    def remove_columns(self, cols: list[str]):
        for x in cols:
            del self[x]

    def items(self):
        return self.data_dict.items()

    def keys(self):
        return self.data_dict.keys()

    def __len__(self):
        return len(self.data_dict[self.ref_key])

    def __getitem__(self, item: Union[str, int, list[int], slice, torch.Tensor]):
        if isinstance(item, str):
            return self.data_dict[item]

        if isinstance(item, list):
            return DictDataset({k: [self.data_dict[k][x] for x in item] for k in self.keys()})

        if isinstance(item, torch.Tensor):
            out_dict = {}
            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    out_dict[k] = v[item]
                else:
                    out_dict[k] = [self.data_dict[k][x.item()] for x in item]
            return DictDataset(out_dict)

        return {k: self.data_dict[k][item] for k in self.keys()}

    def __setitem__(self, key: str, value):
        assert isinstance(key, str)
        assert len(value) == len(self[self.ref_key])
        self.data_dict[key] = value

    def __delitem__(self, key):
        del self.data_dict[key]

        if self.ref_key == key:
            self.ref_key = list(self.keys())[0]

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
