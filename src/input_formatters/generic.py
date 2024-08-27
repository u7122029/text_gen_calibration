from abc import ABC, abstractmethod

from pathlib import Path
from typing import Type, Tuple, Optional

import torch

from calibrators import Calibrator
from data import DictDataset
from utils import dill_load, dill_save, RESULTS_PATH
from llm_models.textgen import TextGenLLMBundle
from prompt_formatters.generic import PromptFormat
from prompt_formatters.cot import CoTPromptFormat
import simple_colors as sc


class InputFormatter(ABC):
    """
    TODO: Determine methods that should be common across all subclasses.
    """

    @abstractmethod
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 dataset: DictDataset,
                 prompt_formatter: PromptFormat,
                 calib_dset_size: Optional[int] = None,
                 test_dset_size: Optional[int] = None):
        """
        Abstract constructor to ensure that this class cannot be instantiated.

        @param llm_bundle: The LLM bundle
        @param dataset: The dataset
        @param correctness_fn: Function to obtain the correctness of each (prediction, answer) pair.
                               It is not the loss function.
        @param calib_dset_size: The size of the calibration set. If this is none along with test_dset_size
        then roughly 70% of the dataset will be in the calibration set.
        @param test_dset_size: The size of the test set. If this is None along with calib_dset_size, then 30%
        of the dataset will appear in the test set.
        If calib_dset_size is not None, then all remaining samples of the provided dataset
                               will be used in the test set.
        """
        if calib_dset_size is not None and test_dset_size is not None:
            assert (calib_dset_size + test_dset_size <= len(dataset),
                    f"size of calibration ({calib_dset_size}) + test dataset ({test_dset_size}) sizes "
                    f"exceed given dataset size.")
        elif calib_dset_size is None and test_dset_size is None:
            calib_dset_size = int(0.7 * len(dataset))
            test_dset_size = len(dataset) - calib_dset_size
        elif calib_dset_size is not None and test_dset_size is None:
            test_dset_size = len(dataset) - calib_dset_size
        else:
            raise Exception("calib_dset_size is None and test_dset_size is not None")

        self.llm_bundle = llm_bundle
        self.dataset = dataset
        self.prompt_formatter = prompt_formatter

        self.target_dir = (Path(RESULTS_PATH) /
                           self.llm_bundle.llm_name /
                           self.__class__.__name__ /
                           self.prompt_formatter.__class__.__name__)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        self.calibrator: Optional[Calibrator] = None

        indices = torch.randperm(len(self.dataset))
        calib_indices = indices[:calib_dset_size]
        test_indices = indices[calib_dset_size: calib_dset_size + test_dset_size]

        self.calib_dataset: DictDataset = self.dataset[calib_indices]
        self.test_dataset: DictDataset = self.dataset[test_indices]

    @abstractmethod
    def get_calibration_and_test_data(self, batch_size=1, recompute=False) -> tuple[DictDataset, DictDataset]:
        pass

    @abstractmethod
    def run_pipeline(self,
                     calibrator_type: Type[Calibrator],
                     batch_size=1,
                     recompute_logits=False,
                     recalibrate=False,
                     **kwargs) -> Tuple[DictDataset, DictDataset]:
        pass

    @abstractmethod
    def perform_calibration(self, calib_data, weights_path, batch_size):
        """
        Calibrate the calibration model. If there already are calibrator weights, we load them in and skip calibration.
        @param calib_data:
        @param weights_path:
        @param batch_size:
        @return:
        """
        pass

    def test_calibrator(self,
                        calibrator_type: Type[Calibrator],
                        original_input_formatter: 'InputFormatter',
                        use_full_dset=True):
        save_path = (original_input_formatter.target_dir /
                     calibrator_type.__name__ /
                     "ood" /
                     f"{self.__class__.__name__}.dill")
        calib_data, test_data = self.get_calibration_and_test_data()
        if use_full_dset:
            test_data = test_data.join(calib_data)
        del calib_data

        # Get the test results.
        if save_path.exists():
            print(f"Getting test results from {save_path}")
            test_results = dill_load(save_path)
        else:
            print(f"Did not find ood test results at {save_path}. Running pipeline.")
            original_input_formatter.run_pipeline(calibrator_type, batch_size=4)

            print("Testing calibrator on test_data")
            test_results = original_input_formatter.calibrator.test(test_data)
            dill_save(test_results, save_path)
        test_data.update(test_results)
        return test_data

    @abstractmethod
    def correctness(self, predictions: list[str], labels: list[str], successful: torch.Tensor):
        pass


class CoTInputFormatter(InputFormatter, ABC):
    @abstractmethod
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 dataset: DictDataset,
                 prompt_formatter: CoTPromptFormat,
                 calib_dset_size=None,
                 test_dset_size=None):
        """
        Abstract constructor to ensure that this class cannot be instantiated.
        """
        InputFormatter.__init__(self, llm_bundle, dataset, prompt_formatter, calib_dset_size, test_dset_size)

        # Format the datasets
        self.numeric_conf_fmt, self.worded_conf_fmt = (
            self.format_verbalised("numeric", "numeric_conf_formatted"),
            self.format_verbalised("worded", "worded_conf_formatted")
        )
        self.calib_dataset.update(self.response_fmt(self.calib_dataset))
        self.test_dataset.update(self.response_fmt(self.test_dataset))
        print(sc.blue(len(self.calib_dataset)))
        print(sc.blue(len(self.test_dataset)))

    def get_calibration_and_test_data(self, batch_size=1, recompute=False):
        """
        Gets the logits and tokens from the llm over the calibration and test datasets.
        No EOS tokens are filtered at all.
        :param batch_size: generation batch size for both calib and test sets.
        :param recompute: whether to recompute the logits for both sets.
        :return:
        """
        print("Getting Calibration and Test data.")
        calib_filepath = self.target_dir / "calib_data"
        test_filepath = self.target_dir / "test_data"

        if calib_filepath.exists() and not recompute:
            print(f"Found existing calibration data in {calib_filepath}")
            calib_conf_dset = dill_load(calib_filepath / "data.dill")
            self.calib_dataset.update(calib_conf_dset)
        else:
            print(f"Calibration data at ({calib_filepath}) not found.")
            self.llm_bundle.load_model()
            with torch.no_grad():
                self.calib_dataset = self.llm_bundle.get_eval_data_from_dset(self.calib_dataset,
                                                                             calib_filepath,
                                                                             batch_size=batch_size,
                                                                             desc="Get Logits + Tokens (Calib)")
            all_predictions, all_predictions_successful = self.prompt_formatter.obtain_answers(
                self.llm_bundle.tokeniser.batch_decode(self.calib_dataset["tokens"])
            )
            self.calib_dataset["correct"] = self.correctness(all_predictions,
                                                             self.calib_dataset["answer"],
                                                             all_predictions_successful)  # here
            self.calib_dataset["pred_successful"] = all_predictions_successful
            self.calib_dataset["prediction"] = all_predictions

            (self.calib_dataset
             .update(self.numeric_conf_fmt(self.calib_dataset))
             .update(self.worded_conf_fmt(self.calib_dataset)))

            with torch.no_grad():
                self.calib_dataset = self.llm_bundle.get_verbalised_confs_from_dset(self.calib_dataset,
                                                                                    batch_size=batch_size,
                                                                                    desc="Get Verbalised Confs (Calib)")

            self.calib_dataset.remove_columns(["response_formatted",
                                               "numeric_conf_formatted",
                                               "worded_conf_formatted"])

            self.calib_dataset.save(calib_filepath / "data.dill")
            print("Calibration data done.")

        if test_filepath.exists() and not recompute:
            print(f"Found existing test data in {test_filepath}")
            test_conf_dset = dill_load(test_filepath / "data.dill")
            print(sc.green(len(self.test_dataset)))
            self.test_dataset.update(test_conf_dset)
        else:
            print(f"test data at ({test_filepath}) not found.")
            self.llm_bundle.load_model()
            with torch.no_grad():
                self.test_dataset = self.llm_bundle.get_eval_data_from_dset(self.test_dataset,
                                                                            test_filepath,
                                                                            batch_size=batch_size,
                                                                            desc="Get Logits + Tokens (Test)")

            all_predictions, all_predictions_successful = self.prompt_formatter.obtain_answers(
                self.llm_bundle.tokeniser.batch_decode(self.test_dataset["tokens"])
            )
            self.test_dataset["correct"] = self.correctness(all_predictions,
                                                            self.test_dataset["answer"],
                                                            all_predictions_successful)
            self.test_dataset["pred_successful"] = all_predictions_successful
            self.test_dataset["prediction"] = all_predictions

            (self.test_dataset
             .update(self.numeric_conf_fmt(self.test_dataset))
             .update(self.worded_conf_fmt(self.test_dataset)))
            with torch.no_grad():
                self.test_dataset = self.llm_bundle.get_verbalised_confs_from_dset(self.test_dataset,
                                                                                   batch_size=batch_size,
                                                                                   desc="Get Verbalised Confs (Test)")

            self.test_dataset.remove_columns(["response_formatted",
                                              "numeric_conf_formatted",
                                              "worded_conf_formatted"])

            self.test_dataset.save(test_filepath / "data.dill")
            print("Test Data done.")

        return self.calib_dataset, self.test_dataset

    def perform_calibration(self, calib_data, weights_path, batch_size):
        cw_path = weights_path / "calib_weights.dill"
        if cw_path.exists():
            print("Loading calibration weights.")
            self.calibrator.load(cw_path)
        else:
            print("Performing calibration of model.")
            weights_path.mkdir(parents=True, exist_ok=True)
            self.calibrator.calibrate(calibration_dset=calib_data,
                                      batch_size=batch_size)
            self.calibrator.save(cw_path)

    def run_pipeline(self,
                     calibrator_type: Type[Calibrator],
                     batch_size=1,
                     recompute_logits=False,
                     recalibrate=False,
                     **kwargs) -> Tuple[DictDataset, DictDataset]:
        """
        The pipeline follows 3 steps.
        1. Get calibration and test data, such as logits, tokens, answers and confidences.
        2. Calibrate the calibrator.
        3. Obtain adjusted confidences from calibration and test sets.
        @param calibrator_type:
        @param batch_size:
        @param recompute_logits:
        @param recalibrate:
        @param kwargs:
        @return:
        """
        calib_data, test_data = self.get_calibration_and_test_data(batch_size,
                                                                   recompute=recompute_logits)

        self.calibrator = calibrator_type(self.llm_bundle)
        weights_path = self.target_dir / self.calibrator.get_name()
        self.perform_calibration(calib_data, weights_path, batch_size)

        # Test the calibrator.
        cr_path = weights_path / "calib_results.dill"
        if cr_path.exists():
            print(f"Found existing calibration results in {cr_path}")
            calib_results = dill_load(cr_path)
        else:
            print(f"Did not find existing calibration results in {cr_path}")
            calib_results = self.calibrator.test(test_dset=calib_data,
                                                 batch_size=batch_size)
            dill_save(calib_results, cr_path)

        tr_path = weights_path / "test_results.dill"
        if tr_path.exists():
            print(f"Found existing test results in {tr_path}")
            test_results = dill_load(tr_path)
        else:
            print(f"Did not find existing test results in {tr_path}")
            test_results = self.calibrator.test(test_dset=test_data,
                                                batch_size=batch_size)
            dill_save(test_results, tr_path)

        calib_data.update(calib_results)
        test_data.update(test_results)

        return calib_data, test_data

    def response_fmt(self, x):
        questions = x['question']
        formatted = []
        for question in questions:
            formatted_q = self.prompt_formatter(question)
            formatted.append(formatted_q)
        return {"response_formatted": formatted}

    def format_verbalised(self, prompt_type, feature_name):
        def format_fn(x):
            questions = x['question']
            preds = x["prediction"]
            formatted = []
            for question, pred in zip(questions, preds):
                formatted_q = self.prompt_formatter.conf_format(question, pred, prompt_type)
                formatted.append(formatted_q)
            return {feature_name: formatted}

        return format_fn
