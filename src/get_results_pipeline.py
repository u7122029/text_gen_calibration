from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC
from torcheval.metrics.functional import binary_auprc

import torch

import fire
from icecream import ic
from calibrators import calibrator_dict
from pathlib import Path
from tabulate import tabulate
from input_formatters import GSMCoT
from data_formats import get_dataset
import os

torch.manual_seed(0)


class CompiledMetrics:
    def __init__(self, confs_before, confs_after, correct, n_bins=15):
        assert len(confs_before) == len(confs_after), "confidences before and after are not the same length."
        assert len(confs_after) == len(correct), "correct is not the same length as confs_after and confs_before."

        self.confs_before = confs_before
        self.confs_after = confs_after
        self.correct = correct

        self.__ece_metric = BinaryCalibrationError(n_bins=n_bins)
        self.__auroc_metric = BinaryAUROC()

        self.ece_before = self.__ece_metric(self.confs_before, self.correct).item()
        self.ece_after = self.__ece_metric(self.confs_after, self.correct).item()

        self.auroc_before = self.__auroc_metric(self.confs_before, self.correct).item()
        self.auroc_after = self.__auroc_metric(self.confs_after, self.correct).item()

        self.auprc_before = binary_auprc(self.confs_before, self.correct).item()
        self.auprc_after = binary_auprc(self.confs_after, self.correct).item()

        self.accuracy = torch.mean(self.correct.float()).item()

        self.confs_diff = self.confs_after - self.confs_before
        self.all_mean_conf_change = torch.mean(self.confs_diff)
        self.correct_mean_conf_change = torch.mean(self.confs_diff[self.correct])
        self.incorrect_mean_conf_change = torch.mean(self.confs_diff[~self.correct])

        self.all_total_conf_change = torch.sum(self.confs_diff)
        self.correct_total_conf_change = torch.sum(self.confs_diff[self.correct])
        self.incorrect_total_conf_change = torch.sum(self.confs_diff[~self.correct])

    def __len__(self):
        return len(self.correct)

    def display(self):
        print(f"No. Samples: {len(self)}")
        print(f"Accuracy: {self.accuracy}")
        print("\nBasic Metrics:")
        table = [
            ["Category",          "ECE",            "AUROC",           "AUPRC"],
            ["Before Calibration", self.ece_before, self.auroc_before, self.auprc_before],
            ["After Calibration",  self.ece_after, self.auroc_after, self.auprc_after]
        ]
        print(tabulate(table[1:], headers=table[0], tablefmt="github"))
        print("\nChanges in Confidences:")
        table1 = [
            ["Category",     "All Preds",                "Correct Preds",               "Incorrect Preds"],
            ["Mean Change",  self.all_mean_conf_change,  self.correct_mean_conf_change,  self.incorrect_mean_conf_change],
            ["Total Change", self.all_total_conf_change, self.correct_total_conf_change, self.incorrect_total_conf_change]
        ]
        print(tabulate(table1[1:], headers=table1[0], tablefmt="github"))

    def save(self, filename):
        out = {
            "confs_before": self.confs_before,
            "confs_after": self.confs_after,
            "correct": self.correct
        }
        torch.save(out, filename)


def show_results(filepath: Path):
    results_dict = torch.load(str(filepath))

    calibrator_name = results_dict["calibrator_name"]
    model_name = results_dict["model_name"]

    calib_metrics = CompiledMetrics(results_dict["calib_confs_original"],
                                    results_dict["calib_confs_calibrated"],
                                    results_dict["calib_correct"])
    test_metrics = CompiledMetrics(results_dict["test_confs_original"],
                                   results_dict["test_confs_calibrated"],
                                   results_dict["test_correct"])

    print(f"Model Name: {model_name}")
    print(f"Calibrator Name: {calibrator_name}")
    terminal_size = os.get_terminal_size().columns
    print("-" * terminal_size)
    print("Calibration Set Results:")
    calib_metrics.display()
    print("-" * terminal_size)
    print("Test Set Results:")
    test_metrics.display()

# HuggingFaceH4/zephyr-7b-beta
# mistralai/Mistral-7B-Instruct-v0.2
# zhengr/MixTAO-7Bx2-MoE-v8.1 cannot use
# google/gemma-1.1-7b-it
# google/gemma-1.1-2b-it
# Qwen/Qwen1.5-1.8B-Chat not good.
# meta-llama/Llama-2-7b-chat-hf
# meta-llama/Meta-Llama-3-8B-Instruct
# 01-ai/Yi-1.5-9B-Chat
# NousResearch/Hermes-2-Theta-Llama-3-8B cannot use
# NousResearch/Hermes-2-Pro-Mistral-7B
def main(prompt_type: str="CoT",
         dataset_name: str="GSM",
         calibrator_type="TemperatureScalingVariant",
         model_name="google/gemma-1.1-2b-it",
         debug_responses=True,
         batch_size=4,
         calib_dset_size=300,
         test_dset_size=300,
         recompute_logits=False,
         retrain_calibrator=False):
    #if prompt_type not in prompt_dict:
    #    raise ValueError(f"prompt_type '{prompt_type}' not in {prompt_dict.keys()}")

    if calibrator_type not in calibrator_dict:
        raise ValueError(f"calibrator_type '{calibrator_type}' not in {calibrator_dict.keys()}")

    # Get token.
    with open("token.txt") as f:
        token = f.read().strip()
        ic(token)

    p = Path("results") / dataset_name / calibrator_type / model_name / prompt_type
    p.parent.mkdir(parents=True, exist_ok=True)
    file_path = Path(f"{str(p)}.pt")
    if file_path.exists() and not retrain_calibrator:
        show_results(file_path)
        quit()

    dataset = get_dataset(dataset_name)
    input_formatter = GSMCoT(model_name, dataset, token, calib_dset_size, test_dset_size)
    calib_confs_original, calib_confs_calibrated, calib_correct = input_formatter.train_calibrator(
        calibrator_dict[calibrator_type],
        batch_size=batch_size,
        recompute_logits=recompute_logits
    )
    test_confs_original, test_confs_calibrated, test_correct = input_formatter.test_calibrator(
        calibrator_dict[calibrator_type]
    )

    compiled = {
        "calib_confs_original": calib_confs_original,
        "calib_confs_calibrated": calib_confs_calibrated,
        "calib_correct": calib_correct,
        "test_confs_original": test_confs_original,
        "test_confs_calibrated": test_confs_calibrated,
        "test_correct": test_correct,
        "model_name": model_name,
        "calibrator_name": calibrator_type,
        "prompt_type": prompt_type,
        #"calibrator_params": None if calibrator is None else calibrator.state_dict()
    }

    torch.save(compiled, f"{str(p)}.pt")
    show_results(file_path)


if __name__ == "__main__":
    fire.Fire(main)
