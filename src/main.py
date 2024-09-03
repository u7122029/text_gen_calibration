import os

import fire
import torch

from calibrators import calibrator_dict
from data import DictDataset
from input_formatters import input_formatter_dict
from llm_models.textgen import TextGenLLMBundle

from metrics import ModelMetrics
from prompt_formatters import PromptVersion


def show_results(calib_data: DictDataset,
                 test_data: DictDataset,
                 model_name: str,
                 calibrator_name: str,
                 input_formatter_name: str):
    details = {"LLM": model_name,
               "Calibrator": calibrator_name,
               "Input Formatter": input_formatter_name}
    calib_results = ModelMetrics(calib_data, **details)
    test_results = ModelMetrics(test_data, **details)
    print("---")
    print("### Calibration Set Results")
    calib_results.display()
    print("---")
    print("### Test Set Results")
    test_results.display()


def main(input_formatter_name: str="GSMCoT",
         calibrator_name="FrequencyTS",
         cot_version="DEFAULT",
         model_name="google/gemma-1.1-2b-it",
         batch_size=4,
         calib_dset_size=None,
         test_dset_size=None,
         recompute_logits=False,
         retrain_calibrator=False):
    torch.manual_seed(0)
    if calibrator_name not in calibrator_dict:
        raise ValueError(f"calibrator_name '{calibrator_name}' not in {calibrator_dict.keys()}")

    llm_bundle = TextGenLLMBundle(model_name)

    cot_version = PromptVersion.from_string(cot_version)
    input_formatter_class = input_formatter_dict[input_formatter_name]
    input_formatter = input_formatter_class(llm_bundle, cot_version, calib_dset_size, test_dset_size)

    calib_data, test_data = input_formatter.run_pipeline(
        calibrator_dict[calibrator_name],
        batch_size,
        recompute_logits=recompute_logits,
        recalibrate=retrain_calibrator
    )

    show_results(calib_data, test_data, model_name, calibrator_name, input_formatter_name)


if __name__ == "__main__":
    fire.Fire(main)
