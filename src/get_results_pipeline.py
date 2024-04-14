from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC, BinaryAccuracy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import re
from datasets import Dataset, tqdm
import fire
from icecream import ic
from chat_formats import prompt_dict
from calibrators import calibrator_dict
from pathlib import Path

torch.manual_seed(0)


def get_dataset(tokeniser, format_chat_func):
    df = pd.read_json("test.jsonl", lines=True)
    df["answer"] = df["answer"].apply(lambda x: int(re.sub(r'[^\w\s]', '', x.split("####")[1])))
    dataset = Dataset.from_pandas(df.iloc[torch.randperm(len(df))[:200].tolist()])
    dataset = dataset.map(lambda x: {"formatted": format_chat_func(x, tokeniser)}, batched=True)
    return dataset


def show_results(filepath: Path, dataset: Dataset):
    ece_metric = BinaryCalibrationError(n_bins=15)
    auroc_metric = BinaryAUROC()
    #accuracy_metric = BinaryAccuracy()

    results_dict = torch.load(str(filepath))
    all_preds = results_dict["all_preds"]
    correct = (all_preds == torch.Tensor(dataset["answer"])).int()
    ic(correct)
    confs_before_calib = results_dict["confs_before_calib"]
    confs_after_calib = results_dict["confs_after_calib"]
    d = {
        "ece_before": ece_metric(confs_before_calib, correct).item(),
        "auroc_before": auroc_metric(confs_before_calib, correct).item(),
        "ece_after": ece_metric(confs_after_calib, correct).item(),
        "auroc_after": auroc_metric(confs_after_calib, correct).item(),
        "acc": torch.mean(correct.float()).item()
    }
    ic(results_dict["model_name"])
    ic(results_dict["prompt_type"])
    ic(results_dict["calibrator_name"])
    ic(results_dict["calibrator_params"])
    ic(d)


# HuggingFaceH4/zephyr-7b-beta
# mistralai/Mistral-7B-Instruct-v0.2
# zhengr/MixTAO-7Bx2-MoE-v8.1
# google/gemma-1.1-7b-it
def main(prompt_type: str="FCoT",
         calibrator_type="ReLu_WATC",
         model_name="mistralai/Mistral-7B-Instruct-v0.2",
         debug_responses=True):
    if prompt_type not in prompt_dict:
        raise ValueError(f"prompt_type '{prompt_type}' not in {prompt_dict.keys()}")

    if calibrator_type not in calibrator_dict:
        raise ValueError(f"calibrator_type '{calibrator_type}' not in {calibrator_dict.keys()}")

    formatter_cls = prompt_dict[prompt_type]

    # Get token.
    with open("token.txt") as f:
        token = f.read()

    tokeniser = AutoTokenizer.from_pretrained(model_name, token=token, padding_side="left")
    tokeniser.pad_token_id = tokeniser.eos_token_id

    dataset = get_dataset(tokeniser, formatter_cls.format_inputs)

    p = Path("results") / calibrator_type / model_name / prompt_type
    p.parent.mkdir(parents=True, exist_ok=True)
    file_path = Path(f"{str(p)}.pt")
    if file_path.exists():
        show_results(file_path,dataset)
        quit()

    dl = DataLoader(dataset, batch_size=2)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 token=token)
                                                 #attn_implementation="flash_attention_2")

    strategy_name = calibrator_dict[calibrator_type]
    strategy = strategy_name(tokeniser, model, debug_responses)
    all_preds, confs_before_calib, confs_after_calib, calibrator = strategy.calibrate(dl, formatter_cls)
    if calibrator is not None:
        ic(list(calibrator.parameters()))

    compiled = {
        #"explanations": all_explanations,
        "all_preds": all_preds,
        "confs_before_calib": confs_before_calib,
        "confs_after_calib": confs_after_calib,
        "model_name": model_name,
        "calibrator_name": calibrator.__class__.__name__,
        "prompt_type": prompt_type,
        "calibrator_params": None if calibrator is None else calibrator.state_dict()
    }

    torch.save(compiled, f"{str(p)}.pt")
    show_results(file_path, dataset)


if __name__ == "__main__":
    fire.Fire(main)
