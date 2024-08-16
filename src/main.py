import os

import fire
import torch

from calibrators import calibrator_dict
from input_formatters import input_formatter_dict
from llm_models.textgen import TextGenLLMBundle

from metrics import ModelMetrics


def show_results(calib_results: ModelMetrics, test_results: ModelMetrics, model_name: str, calibrator_name: str):
    print(f"Model Name: {model_name}")
    print(f"Calibrator Name: {calibrator_name}")
    terminal_size = os.get_terminal_size().columns
    print("-" * terminal_size)
    print("Calibration Set Results:")
    calib_results.display()
    print("-" * terminal_size)
    print("Test Set Results:")
    test_results.display()


def main(input_formatter: str="GSMCoT",
         calibrator_name="TemperatureScaling",
         model_name="google/gemma-1.1-2b-it",
         batch_size=4,
         calib_dset_size=10,
         test_dset_size=10,
         recompute_logits=False,
         retrain_calibrator=False):
    torch.manual_seed(0)
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

    calib_results = ModelMetrics(calib_data, f"{calibrator_name} + {model_name}")
    test_results = ModelMetrics(test_data, f"{calibrator_name} + {model_name}")
    show_results(calib_results, test_results, model_name, calibrator_name)


if __name__ == "__main__":
    fire.Fire(main)
