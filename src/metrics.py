from typing import Optional, Any
from abc import ABC, abstractmethod

from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC
from torchmetrics import Metric
from torcheval.metrics.functional import binary_auprc
import simple_colors as sc

import torch
from tabulate import tabulate
import pandas as pd

from data import DictDataset
from input_formatters import InputFormatter


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
        return torch.sum((self.preds - self.targets)**2) / self.total


class ModelMetrics:
    @abstractmethod
    def __init__(self, data: DictDataset, n_bins=15, **kwargs):
        """

        @param data:
        @param n_bins:
        @param kwargs: Extra details to be included in the results.
        """
        cols = {"logits_confs", "correct", "numeric_confs", "numeric_successful", "worded_confs", "worded_successful",
                "calibrated_confs", "calibrated_successful"}
        keys_to_delete = set(data.keys()) - cols
        for key in keys_to_delete:
            del data[key]

        assert cols.issubset(set(data.keys()))

        self.n_bins = n_bins
        self.extra_details = kwargs
        self.logits_confs = torch.Tensor(data["logits_confs"])
        self.logit_confs_successful = ~self.logits_confs.isnan() # True entries indicate no outputted tokens.

        self.calibrated_confs = torch.Tensor(data["calibrated_confs"])
        self.correct = torch.Tensor(data["correct"]).bool() & self.logit_confs_successful

        self.calibrated_successful = torch.Tensor(data["calibrated_successful"]).bool() & self.logit_confs_successful
        self.calibrated_confs = self.calibrated_confs[self.calibrated_successful]
        self.calibrated_correct = self.correct[self.calibrated_successful]

        # construct verbalised confs
        self.num_success_mask = torch.Tensor(data["numeric_successful"]).bool() & self.logit_confs_successful
        self.worded_success_mask = torch.Tensor(data["worded_successful"]).bool() & ~self.num_success_mask & self.logit_confs_successful
        #self.verbalised_success_mask = (self.num_success_mask | self.worded_success_mask) & self.logit_confs_successful

        self.worded_confs = torch.Tensor(data["worded_confs"])[self.worded_success_mask]
        self.numeric_confs = torch.Tensor(data["numeric_confs"])[self.num_success_mask]

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
        self.verbalised_confs_diff = self.calibrated_confs - self.verbalised_confs[self.calibrated_successful]

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

        self.extra_details.update({
            "No. Samples": len(self),
            "Succeeded VCs": self.num_verbalised_successful,
            "Accuracy": self.accuracy,
            "No. ECE Bins": self.n_bins
        })

    def __len__(self):
        return len(self.correct)

    def display(self):
        print(tabulate(list(self.extra_details.items()), tablefmt="github"))

        print("\n**Basic Metrics:**")
        table = [
            ["Category", "ECE", "Brier", "AUROC", "AUPRC"],
            ["Logit Confs", self.ece_logits, self.brier_logits, self.auroc_logits, self.auprc_logits],
            ["Verbalised Confs", self.ece_verbalised, self.ece_verbalised, self.auroc_verbalised, self.auprc_verbalised],
            ["After Calibration", self.ece_calibrated, self.brier_calibrated, self.auroc_calibrated, self.auprc_calibrated]
        ]
        print(tabulate(table[1:], headers=table[0], tablefmt="github"))
        print("\n**Changes in Confidences:**")
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


class ModelMetricsCollection(list[ModelMetrics]):
    def __init__(self, *args):
        super().__init__(*args)
        self.details = {}

    #def make_details_table(self):
    #    return tabulate(list(self.details.items()), tablefmt="github")

    def generate_tables(self, key: str, control_keys: Optional[list[str]]=None):
        """
        Generate the tables.
        @param key: The name of the independent variable.
        @param control_keys: The keys that are known to be constant in associated value.
        @return:
        """

        table = {
            key: [],
            "accuracy": [],
            "ece_logits": [],
            "ece_verbalised": [],
            "ece_calib": [],
            "brier_logits": [],
            "brier_verbalised": [],
            "brier_calib": [],
            "auroc_logits": [],
            "auroc_verbalised": [],
            "auroc_calib": [],
            "auprc_logits": [],
            "auprc_verbalised": [],
            "auprc_calib": []
        }
        for x in self:
            table[key].append(x.extra_details[key])

            table["accuracy"].append(x.accuracy)

            table["ece_logits"].append(x.ece_logits)
            table["ece_verbalised"].append(x.ece_verbalised),
            table["ece_calib"].append(x.ece_calibrated)

            table["brier_logits"].append(x.brier_logits)
            table["brier_verbalised"].append(x.brier_verbalised)
            table["brier_calib"].append(x.brier_calibrated)

            table["auroc_logits"].append(x.auroc_logits)
            table["auroc_verbalised"].append(x.auroc_verbalised)
            table["auroc_calib"].append(x.auroc_calibrated)

            table["auprc_logits"].append(x.auprc_logits)
            table["auprc_verbalised"].append(x.auprc_verbalised)
            table["auprc_calib"].append(x.auprc_calibrated)
        if control_keys is not None:
            for control_key in control_keys:
                self.details[control_key] = table[control_key][0]
                del table[control_key]
        return pd.DataFrame(table).sort_values("ece_calib")#, list(self.details.items())#tabulate(, tablefmt="github")