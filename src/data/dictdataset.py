from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Union

import dill
import torch
from torch.utils.data import Dataset

from utils import dill_save, dill_load


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

    def save(self, path: Path):
        dill_save(self.data_dict, path)

    def join(self, other: 'DictDataset'):
        common_keys = set(self.keys()).intersection(set(other.keys()))
        out_dict = {}
        for key in common_keys:
            out_dict[key] = self[key, True]
            if isinstance(out_dict[key], list):
                out_dict[key].extend(other[key, True])
            else:
                out_dict[key] = torch.cat([out_dict[key], other[key, True]])
        return DictDataset(out_dict)

    def update(self, other: Union[dict, 'DictDataset']):
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

    def __getitem__(self,
                    item: str |
                          int |
                          list[int] |
                          slice |
                          torch.Tensor |
                          tuple[str | int | list[int] | slice | torch.Tensor, bool]):
        raw = False
        if isinstance(item, tuple):
            raw = item[1]
            item = item[0]

        if isinstance(item, str):
            if isinstance(self.data_dict[item][0], PurePosixPath) and not raw:
                return [dill_load(x) for x in self.data_dict[item]]
            return self.data_dict[item]

        if isinstance(item, list):
            d = {}
            for k, v in self.items():
                if k not in d:
                    d[k] = []

                for x in item:
                    elem = v[x]
                    if isinstance(elem, PurePosixPath):
                        d[k].append(dill_load(elem))
                        continue
                    d[k].append(elem)
            return DictDataset(d)

        if isinstance(item, torch.Tensor):
            out_dict = {}
            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    out_dict[k] = v[item]
                    continue

                if k not in out_dict:
                    out_dict[k] = []

                for x in item:
                    elem = v[x.item()]
                    if isinstance(elem, PurePosixPath):
                        out_dict[k].append(dill_load(elem))
                        continue
                    out_dict[k].append(elem)

            return DictDataset(out_dict)

        # item is of type slice or int
        out_dict = {}
        for k, v in self.items():
            out_dict[k] = v[item]

            if isinstance(item, slice):
                for idx, elem in enumerate(out_dict[k]):
                    if isinstance(elem, PurePosixPath):
                        out_dict[k][idx] = dill_load(elem)
            else:
                if isinstance(out_dict[k], PurePosixPath):
                    out_dict[k] = dill_load(out_dict[k])

        return out_dict

    def __setitem__(self, key: str, value):
        assert isinstance(key, str)
        assert len(value) == len(self[self.ref_key]), f"{len(value)} vs. {len(self[self.ref_key])}."
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
