from typing import Type

import fire
from llm_models.textgen import TextGenLLMBundle
from input_formatters import input_formatter_dict, InputFormatter
from calibrators import calibrator_dict, Calibrator
from main import ModelMetrics
from prompt_formatters import CoTVersion


def main(model_name: str="google/gemma-1.1-2b-it",
         calib_input_formatter_name: str="MATHCoT",
         cot_version: str="DEFAULT",
         calibrator_name: str="FrequencyTS",
         test_input_formatter_name: str="GSMCoT"):
    llm_bundle = TextGenLLMBundle(model_name)
    calib_if: InputFormatter = input_formatter_dict[calib_input_formatter_name](llm_bundle, CoTVersion.from_string(cot_version))
    test_if: InputFormatter = input_formatter_dict[test_input_formatter_name](llm_bundle, CoTVersion.from_string(cot_version))
    calibrator: Type[Calibrator] = calibrator_dict[calibrator_name]
    test_results = test_if.test_calibrator(calibrator, calib_if)
    print(ModelMetrics(test_results, f"{test_input_formatter_name} + {model_name}").display())


if __name__ == "__main__":
    fire.Fire(main)