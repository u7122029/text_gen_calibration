from typing import Optional

import fire
import simple_colors as sc

from input_formatters import input_formatter_dict, InputFormatter
from calibrators import calibrator_dict
from llm_models.textgen import TextGenLLMBundle
from prompt_formatters import PromptVersion
from metrics import ModelMetrics, ModelMetricsCollection
from utils import LossFunc

calibrator_names = ["APRICOT_Original",
                    "TokenCalibrator",
                    "APRICOT_TemperatureScaling",
                    "FrequencyPTS_MSR",
                    "FrequencyPTS_M",
                    "FrequencyPTS_S",
                    "FrequencyPTS_R",
                    "FrequencyPTS_MS",
                    "FrequencyPTS_MR",
                    "FrequencyPTS_SR",
                    "LastHiddenStateCalibrator",
                    "TemperatureScaling",
                    "FrequencyTS_MSR",
                    "FrequencyTS_M",
                    "FrequencyTS_S",
                    "FrequencyTS_R",
                    "FrequencyTS_MR",
                    "FrequencyTS_SR",
                    "FrequencyTS_MS",
                    "APRICOT_FrequencyTS_MSR",
                    "APRICOT_FrequencyTS_M",
                    "APRICOT_FrequencyTS_S",
                    "APRICOT_FrequencyTS_R",
                    "APRICOT_FrequencyTS_MS",
                    "APRICOT_FrequencyTS_SR",
                    "APRICOT_FrequencyTS_MR",
                    "FLHS_MSR",
                    "FLHS_M",
                    "FLHS_S",
                    "FLHS_R",
                    "FLHS_SR",
                    "FLHS_MS",
                    "FLHS_MR",
                    "FLHS_MSR",
                    "APRICOT_FLHS_M",
                    "APRICOT_FLHS_S",
                    "APRICOT_FLHS_R",
                    "APRICOT_FLHS_SR",
                    "APRICOT_FLHS_MS",
                    "APRICOT_FLHS_MR"]


def vary_ood_if(model_name: str, calibrator_name, prompt_version: PromptVersion, id_if_name: str):
    """
    @deprecated
    @param model_name:
    @param calibrator_name:
    @param prompt_version:
    @param id_if_name:
    @return:
    """
    llm_bundle = TextGenLLMBundle(model_name)
    id_if = input_formatter_dict[id_if_name](llm_bundle, prompt_version)
    ood_if_names = set(input_formatter_dict.keys()) - {id_if_name}

    collection = ModelMetricsCollection()
    collection.details = {
        "LLM": model_name,
        "Calibrator": calibrator_name,
        "Prompt Version": prompt_version.name,
        "Calib. Input Formatter": id_if_name
    }
    for ood_if_name in ood_if_names:
        ood_if: InputFormatter = input_formatter_dict[ood_if_name](llm_bundle, prompt_version)
        calibrator_type = calibrator_dict[calibrator_name]
        test_results = ood_if.test_calibrator(calibrator_type, id_if)
        details = {
            "Test Formatter": ood_if_name
        }
        collection.append(ModelMetrics(test_results, **details))
        del test_results
    print(collection.make_details_table())
    print()
    print(collection.generate_tables("Test Formatter").to_markdown(index=False))


def vary_calibrator_ood(model_name: str,
                        prompt_version: PromptVersion,
                        loss_func_name: str,
                        id_if_name: str,
                        ood_if_name: str):
    llm_bundle = TextGenLLMBundle(model_name)
    loss_func = LossFunc.from_string(loss_func_name)

    collection = ModelMetricsCollection()
    collection.details = {
        "LLM": model_name,
        "Prompt Version": prompt_version.name,
        "Loss Function": loss_func_name,
        "Calib. Input Formatter": id_if_name,
        "Test Input Formatter": ood_if_name
    }

    for calibrator_name in calibrator_names:
        calibrator_type = calibrator_dict[calibrator_name]
        id_if = input_formatter_dict[id_if_name](llm_bundle, prompt_version, calibrator_type, loss_func)
        ood_if: InputFormatter = input_formatter_dict[ood_if_name](llm_bundle,
                                                                   prompt_version,
                                                                   calibrator_type,
                                                                   loss_func)

        test_results = ood_if.test_calibrator(id_if)

        details = {"Calibrator": calibrator_name}
        collection.append(ModelMetrics(test_results, **details))
        del test_results

    control_keys = ["accuracy"]
    for name in ["ece", "brier", "auroc", "auprc"]:
        control_keys.extend([f"{name}_logits", f"{name}_verbalised"])
    table, details_tab = collection.generate_tables("Calibrator", control_keys)
    print(details_tab)
    print()
    print(table.sort_values("ece_calib"))


def vary_calibrator_id(model_name: str, loss_func_name: str, prompt_version: PromptVersion, input_formatter: str):
    calib_collection = ModelMetricsCollection()
    calib_collection.details = {
        "Split": "Calibration",
        "LLM": model_name,
        "Prompt Version": prompt_version.name,
        "Input Formatter": input_formatter,
        "Loss Function": loss_func_name
    }

    test_collection = ModelMetricsCollection()
    test_collection.details = {
        "Split": "Test",
        "LLM": model_name,
        "Prompt Version": prompt_version.name,
        "Input Formatter": input_formatter,
        "Loss Function": loss_func_name
    }

    llm_bundle = TextGenLLMBundle(model_name)
    loss_func = LossFunc.from_string(loss_func_name)

    for calibrator_name in calibrator_names:
        print(sc.green(calibrator_name))
        calibrator_type = calibrator_dict[calibrator_name]
        id_if: InputFormatter = input_formatter_dict[input_formatter](llm_bundle,
                                                                      prompt_version,
                                                                      calibrator_type,
                                                                      loss_func)  # NOTE: TEMPORARY DATASET SIZES.
        # run the pipeline to ensure that all the calib and test results have been acquired.
        calib_data, test_data = id_if.run_pipeline(batch_size=4)

        details = {"Calibrator": calibrator_name}
        calib_results = ModelMetrics(calib_data, **details)
        test_results = ModelMetrics(test_data, **details)

        calib_collection.append(calib_results)
        test_collection.append(test_results)

    control_keys = ["accuracy"]
    for name in ["ece", "brier", "auroc", "auprc"]:
        control_keys.extend([f"{name}_logits", f"{name}_verbalised"])
    calib_table, calib_details = calib_collection.generate_tables("Calibrator", control_keys)
    test_table, test_details = test_collection.generate_tables("Calibrator", control_keys)

    print(calib_details)
    print()
    print(calib_table.sort_values("ece_calib"))
    print()
    print(test_details)
    print()
    print(test_table.sort_values("ece_calib"))


def main(model_name: str="microsoft/Phi-3-mini-4k-instruct",
         calibrator_name: str=None,
         loss_func_name: str="CORRECT_AWARE",
         prompt_version: str="DEFAULT",
         id_input_formatter_name: str="SQUADV2CoT",
         ood_input_formatter_name: Optional[str]="GSMCoT"):
    """

    @param model_name:
    @param calibrator_name:
    @param loss_func_name:
    @param prompt_version:
    @param id_input_formatter_name:
    @param ood_input_formatter_name:
    @return:
    """
    prompt_version = PromptVersion.from_string(prompt_version)

    # Check that exactly one of the arguments is None.
    """assert sum([1 if x is None else 0
                for x in [model_name,
                          calibrator_name,
                          loss_func_name,
                          prompt_format,
                          id_input_formatter_name,
                          ood_input_formatter_name]]) == 1"""
    if ood_input_formatter_name is None and calibrator_name is None:
        print("vary_calibrator_id")
        vary_calibrator_id(model_name, loss_func_name, prompt_version, id_input_formatter_name)
    elif ood_input_formatter_name is None:
        print("vary_ood_if")
        vary_ood_if(model_name, calibrator_name, prompt_version, id_input_formatter_name)
    elif calibrator_name is None:
        print("vary_calibrator_ood")
        vary_calibrator_ood(model_name, prompt_version, loss_func_name, id_input_formatter_name, ood_input_formatter_name)


if __name__ == "__main__":
    fire.Fire(main)
