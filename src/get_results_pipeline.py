from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC
from torchmetrics import Metric
from torcheval.metrics.functional import binary_auprc

import torch

import fire
from calibrators import calibrator_dict
from tabulate import tabulate

from data import DictDataset
from input_formatters import input_formatter_dict
import os
from utils import TextGenLLMBundle

torch.manual_seed(0)


class BrierScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=torch.Tensor([]), dist_reduce_fx="sum")
        self.add_state("targets", default=torch.Tensor([]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        self.preds = torch.cat([self.preds, preds])
        self.targets = torch.cat([self.targets, targets])
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        return torch.sum((self.preds - self.targets) ** 2) / self.total


class CompiledMetrics:
    def __init__(self, calib_data: DictDataset, n_bins=15):
        assert "logits_confs" in calib_data
        assert "correct" in calib_data
        assert "numeric_confs" in calib_data
        assert "numeric_successful" in calib_data
        assert "worded_confs" in calib_data
        assert "worded_successful" in calib_data
        assert "calibrated_confs" in calib_data
        assert "calibrated_successful" in calib_data

        self.logits_confs = torch.Tensor(calib_data["logits_confs"])
        self.calibrated_confs = torch.Tensor(calib_data["calibrated_confs"])
        self.correct = torch.Tensor(calib_data["correct"]).bool()

        self.calibrated_successful = torch.Tensor(calib_data["calibrated_successful"]).bool()
        self.calibrated_confs = self.calibrated_confs[self.calibrated_successful]
        self.calibrated_correct = self.correct[self.calibrated_successful]

        # construct verbalised confs
        self.num_success_mask = torch.Tensor(calib_data["numeric_successful"]).bool()
        self.worded_success_mask = torch.Tensor(calib_data["worded_successful"]).bool() & ~self.num_success_mask
        self.verbalised_success_mask = self.num_success_mask | self.worded_success_mask

        self.worded_confs = torch.Tensor(calib_data["worded_confs"])[self.worded_success_mask]
        self.numeric_confs = torch.Tensor(calib_data["numeric_confs"])[self.num_success_mask]

        self.numeric_correct = self.correct[self.num_success_mask]
        self.worded_correct = self.correct[self.worded_success_mask]

        self.verbalised_confs = torch.cat([self.worded_confs, self.numeric_confs])
        self.verbalised_correct = torch.cat([self.worded_correct, self.numeric_correct])
        assert len(self.verbalised_confs) == len(self.verbalised_correct)

        self.num_verbalised_successful = len(self.verbalised_correct)

        self.__ece_metric = BinaryCalibrationError(n_bins=n_bins)
        self.__auroc_metric = BinaryAUROC()
        self.__brier_metric = BrierScore()

        self.ece_logits = self.__ece_metric(self.logits_confs, self.correct).item()
        self.ece_verbalised = self.__ece_metric(self.verbalised_confs, self.verbalised_correct).item()
        self.ece_calibrated = self.__ece_metric(self.calibrated_confs, self.calibrated_correct).item()

        self.auroc_logits = self.__auroc_metric(self.logits_confs, self.correct).item()
        self.auroc_verbalised = self.__auroc_metric(self.verbalised_confs, self.verbalised_correct).item()
        self.auroc_calibrated = self.__auroc_metric(self.calibrated_confs, self.calibrated_correct).item()

        self.auprc_logits = binary_auprc(self.logits_confs, self.correct).item()
        self.auprc_verbalised = binary_auprc(self.verbalised_confs, self.verbalised_correct).item()
        self.auprc_calibrated = binary_auprc(self.calibrated_confs, self.calibrated_correct).item()

        self.brier_logits = self.__brier_metric(self.logits_confs, self.correct).item()
        self.brier_verbalised = self.__brier_metric(self.verbalised_confs, self.verbalised_correct).item()
        self.brier_calibrated = self.__brier_metric(self.calibrated_confs, self.calibrated_correct).item()

        self.accuracy = torch.mean(self.correct.float()).item()

        self.logits_confs_diff = self.calibrated_confs - self.logits_confs[self.calibrated_successful]
        self.verbalised_confs_diff = (self.calibrated_confs[self.verbalised_success_mask[self.calibrated_successful]]
                                      - self.verbalised_confs)

        self.logits_mean_conf_change = torch.mean(self.logits_confs_diff)
        self.correct_logits_mean_conf_change = torch.mean(self.logits_confs_diff[self.correct[self.calibrated_successful]])
        self.incorrect_logits_mean_conf_change = torch.mean(self.logits_confs_diff[~self.correct[self.calibrated_successful]])

        self.verbalised_mean_conf_change = torch.mean(self.verbalised_confs_diff)
        self.correct_verbalised_mean_conf_change = torch.mean(self.verbalised_confs_diff[self.verbalised_correct])
        self.incorrect_verbalised_mean_conf_change = torch.mean(self.verbalised_confs_diff[~self.verbalised_correct])

        self.logits_total_conf_change = torch.sum(self.logits_confs_diff)
        self.correct_logits_total_conf_change = torch.sum(self.logits_confs_diff[self.correct[self.calibrated_successful]])
        self.incorrect_logits_total_conf_change = torch.sum(self.logits_confs_diff[~self.correct[self.calibrated_successful]])

        self.verbalised_total_conf_change = torch.sum(self.verbalised_confs_diff)
        self.correct_verbalised_total_conf_change = torch.sum(self.verbalised_confs_diff[self.verbalised_correct])
        self.incorrect_verbalised_total_conf_change = torch.sum(self.verbalised_confs_diff[~self.verbalised_correct])

    def __len__(self):
        return len(self.correct)

    def display(self):
        print(f"No. Samples: {len(self)}")
        print(f"Accuracy: {self.accuracy}")
        print(f"Number of succeeded verbalised confidences: {self.num_verbalised_successful}")
        print("\nBasic Metrics:")
        table = [
            ["Category", "ECE", "Brier", "AUROC", "AUPRC"],
            ["Logit Confs", self.ece_logits, self.brier_logits, self.auroc_logits, self.auprc_logits],
            ["Verbalised Confs", self.ece_verbalised, self.ece_verbalised, self.auroc_verbalised, self.auprc_verbalised],
            ["After Calibration", self.ece_calibrated, self.brier_calibrated, self.auroc_calibrated, self.auprc_calibrated]
        ]
        print(tabulate(table[1:], headers=table[0], tablefmt="github"))
        print("\nChanges in Confidences:")
        table1 = [
            ["Category", "All Preds", "Correct Preds", "Incorrect Preds"],
            ["Mean Change (Logit Confs)", self.logits_mean_conf_change, self.correct_logits_mean_conf_change,
             self.incorrect_logits_mean_conf_change],
            ["Total Change (Logit Confs)", self.logits_total_conf_change, self.correct_logits_total_conf_change,
             self.incorrect_logits_total_conf_change],
            ["Mean Change (Verbalised Confs)", self.verbalised_mean_conf_change, self.correct_verbalised_mean_conf_change,
            self.incorrect_verbalised_mean_conf_change],
            ["Total Change (Verbalised Confs)", self.verbalised_total_conf_change, self.correct_verbalised_total_conf_change,
             self.incorrect_verbalised_total_conf_change]
        ]
        print(tabulate(table1[1:], headers=table1[0], tablefmt="github"))


def show_results(calib_results: CompiledMetrics, test_results: CompiledMetrics, model_name: str, calibrator_name: str):
    print(f"Model Name: {model_name}")
    print(f"Calibrator Name: {calibrator_name}")
    terminal_size = os.get_terminal_size().columns
    print("-" * terminal_size)
    print("Calibration Set Results:")
    calib_results.display()
    print("-" * terminal_size)
    print("Test Set Results:")
    test_results.display()


# HuggingFaceH4/zephyr-7b-beta
# mistralai/Mistral-7B-Instruct-v0.3
# zhengr/MixTAO-7Bx2-MoE-v8.1 cannot use
# google/gemma-1.1-7b-it
# google/gemma-1.1-2b-it
# Qwen/Qwen1.5-1.8B-Chat not good.
# meta-llama/Llama-2-7b-chat-hf
# meta-llama/Meta-Llama-3-8B-Instruct
# 01-ai/Yi-1.5-9B-Chat
# NousResearch/Hermes-2-Theta-Llama-3-8B cannot use
# NousResearch/Hermes-2-Pro-Mistral-7B
# microsoft/Phi-3-mini-4k-instruct
def main(input_formatter: str="GSMCoT",
         calibrator_name="MeanLogitConfsPlattScaling",
         model_name="google/gemma-1.1-2b-it",
         batch_size=4,
         calib_dset_size=300,
         test_dset_size=300,
         recompute_logits=False,
         retrain_calibrator=False):
    if calibrator_name not in calibrator_dict:
        raise ValueError(f"calibrator_name '{calibrator_name}' not in {calibrator_dict.keys()}")

    llm_bundle = TextGenLLMBundle(model_name)
    input_formatter_class = input_formatter_dict[input_formatter]
    input_formatter = input_formatter_class(llm_bundle, calib_dset_size, test_dset_size)

    calib_data, test_data = input_formatter.run_calibration_pipeline(
        calibrator_dict[calibrator_name],
        batch_size,
        recompute_logits=recompute_logits,
        recalibrate=retrain_calibrator
    )

    calib_results = CompiledMetrics(calib_data)
    test_results = CompiledMetrics(test_data)
    show_results(calib_results, test_results, model_name, calibrator_name)


if __name__ == "__main__":
    fire.Fire(main)
