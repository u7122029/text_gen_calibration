from pathlib import Path

import torch
from torch.utils.data import Dataset

from data import DictDataset
from utils import dill_load


class FolderDataset(Dataset):
    def __init__(self, root: Path):
        self.root = root
        self.filenames = list(self.root.glob("*.dill"))

    def to_dictdataset(self, *keys):
        out_dict = {}
        to_tensor_keys = set()
        for x in self:
            for key in keys:
                if key not in out_dict:
                    out_dict[key] = []
                item = x[key]
                if isinstance(item, torch.Tensor) and item.ndim == 0:
                    to_tensor_keys.add(key)
                out_dict[key].append(item)

        for key in to_tensor_keys:
            out_dict[key] = torch.Tensor(out_dict[key])
        return DictDataset(out_dict)

    def __getitem__(self, inp):
        if isinstance(inp, int):
            return dill_load(self.filenames[inp])

        if isinstance(inp, slice):
            items = [dill_load(x) for x in self.filenames[inp]]
        else:
            items = [dill_load(self.filenames[i]) for i in inp]

        assert len(inp) > 0
        out_dict = {}
        for x in items:
            for k, v in x.items():
                if k not in out_dict:
                    out_dict[k] = []
                out_dict[k].append(v)

        for k in out_dict.keys():
            q = out_dict[k]
            if isinstance(q, torch.Tensor) and q.ndim == 0:
                out_dict[k] = torch.Tensor(q)

        return out_dict

    def __len__(self):
        return len(self.filenames)

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