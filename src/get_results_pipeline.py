from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC
from torcheval.metrics.functional import binary_auprc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import re
from datasets import Dataset
import fire
from icecream import ic
from chat_formats import prompt_dict
from calibrators import calibrator_dict
from pathlib import Path
from chat_formats import CoT
from tabulate import tabulate

torch.manual_seed(0)


def get_dataset(tokeniser, format_chat_func, size=None):
    df = pd.read_json("test.jsonl", lines=True)
    df["answer"] = df["answer"].apply(lambda x: int(re.sub(r'[^\w\s]', '', x.split("####")[1])))
    if size is not None:
        dataset = Dataset.from_pandas(df.iloc[torch.randperm(len(df))[:size].tolist()])
    else:
        dataset = Dataset.from_pandas(df.iloc[torch.randperm(len(df)).tolist()])

    if format_chat_func is None:
        dataset = dataset.map(lambda x: {"formatted": x["question"]}, batched=True)
    else:
        dataset = dataset.map(lambda x: {"formatted": format_chat_func(x, tokeniser)}, batched=True)
    return dataset


def show_results(filepath: Path, dataset: Dataset):
    ece_metric = BinaryCalibrationError(n_bins=15)
    auroc_metric = BinaryAUROC()

    results_dict = torch.load(str(filepath))
    all_preds = results_dict["all_preds"]
    correct = (all_preds == torch.Tensor(dataset["answer"])).int()
    correct_bool = correct.bool()

    confs_before_calib = results_dict["confs_before_calib"]
    confs_after_calib = results_dict["confs_after_calib"]
    confs_diff = confs_after_calib - confs_before_calib
    ic(torch.where(torch.isnan(confs_before_calib))[0])
    ic(torch.where(torch.isnan(confs_after_calib))[0])
    d = {
        "ece_before":   ece_metric(confs_before_calib, correct).item(),
        "auroc_before": auroc_metric(confs_before_calib, correct).item(),
        "auprc_before": binary_auprc(confs_before_calib, correct).item(),
        "ece_after":    ece_metric(confs_after_calib, correct).item(),
        "auroc_after":  auroc_metric(confs_after_calib, correct).item(),
        "auprc_after":  binary_auprc(confs_after_calib, correct).item(),
        "acc":          torch.mean(correct.float()).item()
    }
    num_samples = len(correct)
    accuracy = d["acc"]
    ic(num_samples)
    ic(accuracy)
    ic("Basic Metrics:")
    table = [["Category",           "ECE",           "AUROC",           "AUPRC"],
             ["Before Calibration", d["ece_before"], d["auroc_before"], d["auprc_before"]],
             ["After Calibration",  d["ece_after"],  d["auroc_after"],  d["auprc_after"]]
            ]
    print(tabulate(table[1:], headers=table[0], tablefmt="heavy_outline"))
    ic("Changes in Confidences:")
    table1 = [["Category",     "All Preds",            "Correct Preds",                      "Incorrect Preds"],
              ["Mean Change",  torch.mean(confs_diff), torch.mean(confs_diff[correct_bool]), torch.mean(confs_diff[~correct_bool])],
              ["Total Change", torch.sum(confs_diff),  torch.sum(confs_diff[correct_bool]),  torch.sum(confs_diff[~correct_bool])]
             ]
    print(tabulate(table1[1:], headers=table1[0], tablefmt="heavy_outline"))

# HuggingFaceH4/zephyr-7b-beta
# mistralai/Mistral-7B-Instruct-v0.2
# zhengr/MixTAO-7Bx2-MoE-v8.1
# google/gemma-1.1-7b-it
# google/gemma-1.1-2b-it
# Qwen/Qwen1.5-1.8B-Chat
# meta-llama/Llama-2-7b-chat-hf
# meta-llama/Meta-Llama-3-8B-Instruct
def main(prompt_type: str="CoT",
         calibrator_type="TemperatureScalingVariant",
         model_name="google/gemma-1.1-2b-it",
         debug_responses=True,
         redo_results=True):
    if prompt_type not in prompt_dict:
        raise ValueError(f"prompt_type '{prompt_type}' not in {prompt_dict.keys()}")

    if calibrator_type not in calibrator_dict:
        raise ValueError(f"calibrator_type '{calibrator_type}' not in {calibrator_dict.keys()}")

    formatter_cls = prompt_dict[prompt_type]

    # Get token.
    with open("token.txt") as f:
        token = f.read().strip()
        ic(token)

    tokeniser = AutoTokenizer.from_pretrained(model_name, token=token, padding_side="left")
    tokeniser.pad_token_id = tokeniser.eos_token_id

    dataset = get_dataset(tokeniser,
                          lambda x, y: formatter_cls.format_inputs(x,
                                                                   y,
                                                                   template_type=CoT.ChatTemplateType.USER_CHAT),
                          750)

    p = Path("results") / calibrator_type / model_name / prompt_type
    p.parent.mkdir(parents=True, exist_ok=True)
    file_path = Path(f"{str(p)}.pt")
    if file_path.exists() and not redo_results:
        show_results(file_path,dataset)
        quit()

    dl = DataLoader(dataset, batch_size=4)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 token=token)
                                                 #attn_implementation="flash_attention_2")

    strategy_name = calibrator_dict[calibrator_type]
    strategy = strategy_name(tokeniser, model, debug_responses)
    all_preds, confs_before_calib, confs_after_calib, calibrator = strategy.calibrate(dl, formatter_cls)

    compiled = {
        #"explanations": all_explanations,
        "all_preds": all_preds,
        "confs_before_calib": confs_before_calib,
        "confs_after_calib": confs_after_calib,
        "model_name": model_name,
        "calibrator_name": calibrator.__class__.__name__,
        "prompt_type": prompt_type,
        #"calibrator_params": None if calibrator is None else calibrator.state_dict()
    }

    torch.save(compiled, f"{str(p)}.pt")
    show_results(file_path, dataset)


if __name__ == "__main__":
    fire.Fire(main)
