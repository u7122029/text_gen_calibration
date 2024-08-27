from typing import Optional

import fire
from input_formatters import input_formatter_dict, InputFormatter
from calibrators import calibrator_dict
from llm_models.textgen import TextGenLLMBundle
from prompt_formatters import PromptVersion
import simple_colors as sc
from metrics import ModelMetrics, ModelMetricsCollection


def vary_ood_if(model_name: str, calibrator_name, prompt_version: PromptVersion, id_if_name: str):
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


def vary_calibrator(model_name: str, prompt_version: PromptVersion, id_if_name: str, ood_if_name: str):
    llm_bundle = TextGenLLMBundle(model_name)
    id_if = input_formatter_dict[id_if_name](llm_bundle, prompt_version)

    calibrator_names = ["TemperatureScaling", "FrequencyTS", "FrequencyTSTopOnly", "FrequencyTSBotOnly", "FrequencyTSMeanOnly", "FrequencyTSMeanStdOnly", "FrequencyTSNoRF"]

    collection = ModelMetricsCollection()
    collection.details = {
        "LLM": model_name,
        "Prompt Version": prompt_version.name,
        "Calib. Input Formatter": id_if_name,
        "Test Input Formatter": ood_if_name
    }
    for calibrator_name in calibrator_names:
        ood_if: InputFormatter = input_formatter_dict[ood_if_name](llm_bundle, prompt_version)
        print(sc.red(calibrator_name))
        calibrator_type = calibrator_dict[calibrator_name]
        test_results = ood_if.test_calibrator(calibrator_type, id_if)

        details = {"Calibrator": calibrator_name}
        collection.append(ModelMetrics(test_results, **details))
        del test_results
    print(collection.make_details_table())
    print()
    print(collection.generate_tables("Calibrator").to_markdown(index=False))


def main(model_name: str="google/gemma-1.1-2b-it",
         calibrator_name: str=None,
         prompt_version: str="DEFAULT",
         id_input_formatter_name: str="GSMCoT",
         ood_input_formatter_name: Optional[str]="MATHCoT"):
    """

    @param model_name:
    @param calibrator_name:
    @param id_input_formatter_name:
    @param ood_input_formatter_name:
    @return:
    """
    prompt_version = PromptVersion.from_string(prompt_version)

    # Check that exactly one of the arguments is None.
    assert sum([1 if x is None else 0
                for x in [model_name,
                          calibrator_name,
                          prompt_version,
                          id_input_formatter_name,
                          ood_input_formatter_name]]) == 1

    if ood_input_formatter_name is None:
        vary_ood_if(model_name, calibrator_name, prompt_version, id_input_formatter_name)
    elif calibrator_name is None:
        vary_calibrator(model_name, prompt_version, id_input_formatter_name, ood_input_formatter_name)

    """input_formatter: InputFormatter = input_formatter_dict[id_input_formatter_name]
    results_root = Path(RESULTS_PATH)
    metric_results_calib = ModelMetricsCollection()
    metric_results_test = ModelMetricsCollection()

    llm_bundle = TextGenLLMBundle(model_name)
    input_formatter = input_formatter(llm_bundle, 300, 300)
    for calibrator_name in ['TemperatureScaling',
                            'FrequencyTS',
                            'FrequencyTSBotOnly',
                            'FrequencyTSMeanOnly',
                            'FrequencyTSMeanStdOnly',
                            'FrequencyTSNoRF',
                            'FrequencyTSTopOnly']:
        calibrator = calibrator_dict[calibrator_name]
        calib_data, test_data = input_formatter.run_pipeline(calibrator)
        results_dir = results_root / model_name / input_formatter.__class__.__name__ / calibrator_name
        calib_results = dill_load(results_dir / "calib_results.dill")
        test_results = dill_load(results_dir / "test_results.dill")
        calib_data.update(calib_results)
        test_results.update(test_results)

        metric_results_calib.append(ModelMetrics(calib_data, calibrator_name))
        metric_results_test.append(ModelMetrics(test_data, calibrator_name))

    #for metric_result in metric_results_calib:
    #    print(len(metric_result))
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    with open("temp_output.txt", "w") as f:
        f.write(metric_results_calib.generate_tables().to_markdown(index=False) + "\n\n")
        f.write(metric_results_test.generate_tables().to_markdown(index=False))"""


if __name__ == "__main__":
    fire.Fire(main)
