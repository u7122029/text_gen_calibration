from abc import ABC, abstractmethod
from copy import deepcopy

from pathlib import Path
from typing import Type, Tuple, Optional

import torch

from calibrators import Calibrator
from data import DictDataset
from prompt_formatters import PromptVersion
from utils import dill_load, dill_save, RESULTS_PATH, LossFunc
from llm_models.textgen import TextGenLLMBundle
from prompt_formatters.generic import PromptFormat
import simple_colors as sc


class InputFormatter(ABC):
    @abstractmethod
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 dataset: DictDataset,
                 prompt_version: PromptVersion,
                 calibrator_type: Type[Calibrator],
                 loss_fn: LossFunc,
                 calib_dset_size: Optional[int] = None,
                 test_dset_size: Optional[int] = None,
                 _pf_variant: str = None,
                 _mcq_options: set[str] = None):
        """
        Abstract constructor to ensure that this class cannot be instantiated.

        @param llm_bundle: The LLM bundle
        @param dataset: The dataset
        @param prompt_version:
        @param calibrator_type:
        @param loss_fn:
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

        self.__llm_bundle = llm_bundle
        self.__dataset = dataset
        self.__prompt_formatter = prompt_version(variant=_pf_variant)(llm_bundle, mcq_options=_mcq_options)
        self.__loss_fn = loss_fn

        self.__logits_dir = (Path(RESULTS_PATH) /
                             self.llm_bundle.llm_name /
                             self.__class__.__name__ /
                             prompt_version.name)

        self.logits_dir.mkdir(parents=True, exist_ok=True)

        self.calibrator_type: Type[Calibrator] = calibrator_type
        #self.__calibrator: Optional[Calibrator] = self.calibrator_type(self.llm_bundle, self.loss_fn())

        # reset seed to get same indices
        torch.manual_seed(0)
        indices = torch.randperm(len(self.dataset))
        calib_indices = indices[:calib_dset_size]
        test_indices = indices[calib_dset_size: calib_dset_size + test_dset_size]

        self.__calib_dataset: DictDataset = self.dataset[calib_indices]
        self.__test_dataset: DictDataset = self.dataset[test_indices]

    """
    Properties below make the corresponding attributes read-only while still being accessible through child classes.
    """

    @property
    def calib_dataset(self):
        return self.__calib_dataset

    @property
    def test_dataset(self):
        return self.__test_dataset

    @property
    def logits_dir(self):
        return self.__logits_dir

    @property
    def calibrator_dir(self):
        return self.__calibrator_dir

    @property
    def llm_bundle(self):
        return self.__llm_bundle

    @property
    def dataset(self):
        return self.__dataset

    @property
    def prompt_formatter(self):
        return self.__prompt_formatter

    @property
    def calibrator_type(self):
        return self.__calibrator_type

    @calibrator_type.setter
    def calibrator_type(self, new_calibrator: Type[Calibrator]):
        self.__calibrator_type = new_calibrator
        self.__calibrator_dir = self.__logits_dir / self.loss_fn.name / self.__calibrator_type.__name__

        self.__calibrator_dir.mkdir(parents=True, exist_ok=True)

    #@property
    #def calibrator(self):
    #    return self.__calibrator

    @property
    def loss_fn(self):
        return self.__loss_fn

    @abstractmethod
    def get_calib_test_data(self, batch_size=1, recompute=False) -> tuple[DictDataset, DictDataset]:
        pass

    @abstractmethod
    def run_pipeline(self,
                     batch_size=1,
                     recompute_logits=False,
                     recalibrate=False,
                     **kwargs) -> Tuple[DictDataset, DictDataset]:
        pass

    @abstractmethod
    def perform_calibration(self, calibrator, calib_data, weights_path, batch_size):
        """
        Calibrate the calibration model. If there already are calibrator weights, we load them in and skip calibration.
        @param calibrator:
        @param calib_data:
        @param weights_path:
        @param batch_size:
        @return:
        """
        pass

    def test_calibrator(self,
                        original_input_formatter: 'InputFormatter',
                        batch_size=4,
                        use_full_dset=True):
        """
        Test the calibrator from another input formatter on this one.
        @param original_input_formatter: The original input formatter
        @param batch_size: The batch size.
        @param use_full_dset: Whether the full dataset should be used.
        @return:
        """
        self.llm_bundle.load_model(silent=True, lm_head_only=True)
        save_path = (original_input_formatter.calibrator_dir /
                     "ood" /
                     f"{self.__class__.__name__}.dill")
        calib_data, test_data = self.get_calib_test_data(batch_size)
        accuracy = torch.mean(calib_data["correct"].float())
        if use_full_dset:
            test_data = deepcopy(test_data).join(calib_data)
        del calib_data

        # Get the test results.
        if save_path.exists():
            print(f"Getting test results from {save_path}")
            test_results = dill_load(save_path)
        else:
            print(f"Did not find ood test results at {save_path}. Running pipeline.")
            original_input_formatter.run_pipeline(batch_size=batch_size)

            calibrator = original_input_formatter.calibrator_type(original_input_formatter.llm_bundle,
                                                                  original_input_formatter.loss_fn(weight=accuracy))
            calibrator.load(original_input_formatter.calibrator_dir / "calib_weights.dill")

            print("Testing calibrator on test_data")
            test_results = calibrator.test(test_data)
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
                 prompt_version: PromptVersion,
                 calibrator_type: Type[Calibrator],
                 loss_fn: LossFunc,
                 calib_dset_size: Optional[int] = None,
                 test_dset_size: Optional[int] = None,
                 _pf_variant: str = None,
                 _mcq_options: set[str] = None):
        """

        @param llm_bundle:
        @param dataset:
        @param prompt_version:
        @param calibrator_type:
        @param loss_fn:
        @param calib_dset_size:
        @param test_dset_size:
        """
        InputFormatter.__init__(self,
                                llm_bundle,
                                dataset,
                                prompt_version,
                                calibrator_type,
                                loss_fn,
                                calib_dset_size,
                                test_dset_size,
                                _pf_variant=_pf_variant,
                                _mcq_options=_mcq_options)

        # Format the datasets
        self.numeric_conf_fmt, self.worded_conf_fmt = (
            self.format_verbalised("numeric", "numeric_conf_formatted"),
            self.format_verbalised("worded", "worded_conf_formatted")
        )
        self.calib_dataset.update(self.response_fmt(self.calib_dataset))
        self.test_dataset.update(self.response_fmt(self.test_dataset))

    def _get_data(self, dset: DictDataset, save_root: Path, version: str = "Calib", batch_size=4, recompute=False):
        if (save_root / "data.dill").exists() and not recompute:
            print(f"Found existing data in {save_root}")
            conf_dset = dill_load(save_root / "data.dill")
            dset.update(conf_dset)
        else:
            print(f"Data at ({save_root}) not found.")
            with torch.no_grad():
                dset = self.llm_bundle.get_eval_data_from_dset(dset,
                                                               save_root,
                                                               batch_size=batch_size,
                                                               desc=f"Get Logits + Tokens ({version})")
            all_predictions, all_predictions_successful = self.prompt_formatter.obtain_answers(
                self.llm_bundle.tokeniser.batch_decode(dset["tokens"], skip_special_tokens=True)
            )
            dset["correct"] = self.correctness(all_predictions,
                                               dset["answer"],
                                               all_predictions_successful)
            dset["pred_successful"] = all_predictions_successful
            dset["prediction"] = all_predictions

            dset.update(self.numeric_conf_fmt(dset)).update(self.worded_conf_fmt(dset))

            with torch.no_grad():
                dset = self.llm_bundle.get_verbalised_confs_from_dset(dset,
                                                                      batch_size=batch_size,
                                                                      desc=f"Get Verbalised Confs ({version})")

            dset.remove_columns(["response_formatted",
                                 "numeric_conf_formatted",
                                 "worded_conf_formatted"])

            dset.save(save_root / "data.dill")
        return dset

    def get_calib_test_data(self, batch_size=1, recompute=False):
        """
        Gets the logits and tokens from the llm over the calibration and test datasets.
        No EOS tokens are filtered at all.
        :param batch_size: generation batch size for both calib and test sets.
        :param recompute: whether to recompute the logits for both sets.
        :return:
        """
        print("Getting Calibration and Test data.")
        calib_filepath = self.logits_dir / "calib_data"
        test_filepath = self.logits_dir / "test_data"

        calib_dataset = self._get_data(self.calib_dataset, calib_filepath, "Calib", batch_size, recompute)
        print("Calibration data done.")

        test_dataset = self._get_data(self.test_dataset, test_filepath, "Test", batch_size, recompute)
        print("Test data done.")

        self.llm_bundle.unload_model()
        return calib_dataset, test_dataset

    def perform_calibration(self, calibrator, calib_data, weights_path, batch_size):
        cw_path = weights_path / "calib_weights.dill"
        if cw_path.exists():
            print("Loading calibration weights.")
            calibrator.load(cw_path)
        else:
            print("Performing calibration of model.")
            weights_path.mkdir(parents=True, exist_ok=True)
            calibrator.calibrate(calibration_dset=calib_data,
                                 batch_size=batch_size)
            calibrator.save(cw_path)

    def run_pipeline(self,
                     batch_size=4,
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
        @param loss_fn:
        @param recompute_logits:
        @param recalibrate:
        @param kwargs:
        @return:
        """
        calib_data, test_data = self.get_calib_test_data(batch_size,
                                                         recompute=recompute_logits)

        weights_path = self.calibrator_dir
        calibrator = self.calibrator_type(self.llm_bundle,
                                          self.loss_fn(weight=torch.mean(calib_data["correct"].float())))
        self.perform_calibration(calibrator, calib_data, weights_path, batch_size)

        # Test the calibrator.
        calib_results_path = weights_path / "calib_results.dill"
        if calib_results_path.exists():
            print(f"Found existing calibration results in {calib_results_path}")
            calib_results = dill_load(calib_results_path)
        else:
            print(f"Did not find existing calibration results in {calib_results_path}")
            self.llm_bundle.load_model(silent=True, lm_head_only=True)
            self.llm_bundle.unload_model()
            calib_results = calibrator.test(test_dset=calib_data,
                                            batch_size=batch_size)
            dill_save(calib_results, calib_results_path)

        test_results_path = weights_path / "test_results.dill"
        if test_results_path.exists():
            print(f"Found existing test results in {test_results_path}")
            test_results = dill_load(test_results_path)
        else:
            print(f"Did not find existing test results in {test_results_path}")
            self.llm_bundle.load_model(silent=True, lm_head_only=True)
            self.llm_bundle.unload_model()
            test_results = calibrator.test(test_dset=test_data,
                                           batch_size=batch_size)
            dill_save(test_results, test_results_path)

        calib_data.update(calib_results)
        test_data.update(test_results)

        return calib_data, test_data

    def response_fmt(self, x):
        questions = x['question']
        contexts = [None] * len(questions)
        if "context" in x:
            contexts = x["context"]

        formatted = []
        for question, context in zip(questions, contexts):
            formatted_q = self.prompt_formatter(question, context)
            formatted.append(formatted_q)
        return {"response_formatted": formatted}

    def format_verbalised(self, prompt_type, feature_name):
        def format_fn(x):
            questions = x['question']
            preds = x["prediction"]

            contexts = [None] * len(questions)
            if "context" in x:
                contexts = x["context"]

            formatted = []
            for question, context, pred in zip(questions, contexts, preds):
                formatted_q = self.prompt_formatter.conf_format(question, context, pred, prompt_type)
                formatted.append(formatted_q)
            return {feature_name: formatted}

        return format_fn
