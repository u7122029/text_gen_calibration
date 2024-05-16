from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC
from torcheval.metrics.functional import binary_auprc

import torch

import fire
from icecream import ic
from chat_formats import prompt_dict
from calibrators import calibrator_dict
from pathlib import Path
from tabulate import tabulate
from input_formatters import GSMCoT

torch.manual_seed(0)


def show_results(filepath: Path):
    ece_metric = BinaryCalibrationError(n_bins=15)
    auroc_metric = BinaryAUROC()

    results_dict = torch.load(str(filepath))
    correct = results_dict["correct"]
    calibrator_name = results_dict["calibrator_name"]
    model_name = results_dict["model_name"]

    confs_before_calib = results_dict["confs_before_calib"]
    confs_after_calib = results_dict["confs_after_calib"]
    confs_diff = confs_after_calib - confs_before_calib
    ic(torch.where(torch.isnan(confs_before_calib))[0])
    ic(torch.where(torch.isnan(confs_after_calib))[0])
    d = {
        "ece_before": ece_metric(confs_before_calib, correct).item(),
        "auroc_before": auroc_metric(confs_before_calib, correct).item(),
        "auprc_before": binary_auprc(confs_before_calib, correct).item(),
        "ece_after": ece_metric(confs_after_calib, correct).item(),
        "auroc_after": auroc_metric(confs_after_calib, correct).item(),
        "auprc_after": binary_auprc(confs_after_calib, correct).item(),
        "acc": torch.mean(correct.float()).item()
    }
    num_samples = len(correct)
    accuracy = d["acc"]
    print(f"Model Name: {model_name}")
    print(f"Calibrator Name: {calibrator_name}")
    print(f"No. Samples: {num_samples}")
    print(f"Accuracy: {accuracy}")
    print("Basic Metrics:")
    table = [["Category", "ECE", "AUROC", "AUPRC"],
             ["Before Calibration", d["ece_before"], d["auroc_before"], d["auprc_before"]],
             ["After Calibration", d["ece_after"], d["auroc_after"], d["auprc_after"]]
             ]
    print(tabulate(table[1:], headers=table[0], tablefmt="heavy_outline"))
    print("Changes in Confidences:")
    table1 = [["Category", "All Preds", "Correct Preds", "Incorrect Preds"],
              ["Mean Change", torch.mean(confs_diff), torch.mean(confs_diff[correct]),
               torch.mean(confs_diff[~correct])],
              ["Total Change", torch.sum(confs_diff), torch.sum(confs_diff[correct]), torch.sum(confs_diff[~correct])]
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
         dataset_name: str="GSM",
         calibrator_type="TemperatureScalingVariant",
         model_name="google/gemma-1.1-2b-it",
         debug_responses=True,
         redo_results=True,
         batch_size=4,
         dset_size=300):
    if prompt_type not in prompt_dict:
        raise ValueError(f"prompt_type '{prompt_type}' not in {prompt_dict.keys()}")

    if calibrator_type not in calibrator_dict:
        raise ValueError(f"calibrator_type '{calibrator_type}' not in {calibrator_dict.keys()}")

    # Get token.
    with open("token.txt") as f:
        token = f.read().strip()
        ic(token)

    p = Path("results") / dataset_name / calibrator_type / model_name / prompt_type
    p.parent.mkdir(parents=True, exist_ok=True)
    file_path = Path(f"{str(p)}.pt")
    if file_path.exists() and not redo_results:
        show_results(file_path)
        quit()

    input_formatter = GSMCoT(model_name, token, dset_size)
    confs_before_calib, confs_after_calib, correct = input_formatter.apply_calibrator(
        calibrator_dict[calibrator_type],
        results_batch_size=batch_size,
        calibration_batch_size=batch_size
    )
    """    
    dl = DataLoader(dataset, batch_size=batch_size)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 token=token)
                                                 #attn_implementation="flash_attention_2")

    strategy_name = calibrator_dict[calibrator_type]
    strategy = strategy_name(tokeniser, model, debug_responses)
    all_preds, confs_before_calib, confs_after_calib, calibrator = strategy.calibrate(dl, formatter_cls)"""

    compiled = {
        #"explanations": all_explanations,
        #"all_preds": all_preds,
        "confs_before_calib": confs_before_calib,
        "confs_after_calib": confs_after_calib,
        "correct": correct,
        "model_name": model_name,
        "calibrator_name": calibrator_type,
        "prompt_type": prompt_type,
        #"calibrator_params": None if calibrator is None else calibrator.state_dict()
    }

    torch.save(compiled, f"{str(p)}.pt")
    show_results(file_path)


if __name__ == "__main__":
    fire.Fire(main)
