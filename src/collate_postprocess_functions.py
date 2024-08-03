import torch


def logit_token_repeat_label_key(label_key):
    def collate_fn(out_dict: dict):
        out_dict["logits"] = torch.cat(out_dict["logits"], dim=0)
        out_dict[label_key] = torch.cat(
            [c.repeat(len(t)) for c, t in zip(out_dict[label_key], out_dict["tokens"])]).float()
        out_dict["tokens"] = torch.cat(out_dict["tokens"])
        return out_dict
    return collate_fn


def postprocess_target_confs(out_dict):
    out_dict["target_confs"] = torch.Tensor(out_dict["target_confs"])
    return out_dict
