from pathlib import Path
from typing import Optional

import fire
import pandas as pd
import simple_colors as sc
from tabulate import tabulate

from input_formatters import input_formatter_dict, InputFormatter
from calibrators import calibrator_dict
from llm_models.textgen import TextGenLLMBundle
from prompt_formatters import PromptVersion
from metrics import ModelMetrics, ModelMetricsCollection
from utils import LossFunc, RESULTS_PATH

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
        print(sc.green(calibrator_name))
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

    return collection
    """control_keys = ["accuracy"]
    for name in ["ece", "brier", "auroc", "auprc"]:
        control_keys.extend([f"{name}_logits", f"{name}_verbalised"])

    table = collection.generate_tables("Calibrator", control_keys)
    print(tabulate(collection.details.items(), tablefmt="github"))
    print()
    print(table)"""


def merge_dfs(df1, df2):
    # Merge the dataframes on 'name'
    merged = pd.merge(df1, df2, on='Calibrator', suffixes=('_1', '_2'), how='left')

    # Create a mask for rows where df2's score is smaller (or df1's score is NaN)
    mask = (merged['ece_calib_2'] < merged['ece_calib_1']) | (merged['ece_calib_1'].isna())

    # Update result where the mask is True
    cols = ["ece_calib", "brier_calib", "auroc_calib", "auprc_calib", "loss_fn"]
    for col in cols:
        df1.loc[mask, col] = merged.loc[mask, f"{col}_2"]
    #df1.loc[mask, 'loss_fn'] = merged.loc[mask, 'loss_fn_2']

    return df1


def compare_collections_by_loss(collections: list[ModelMetricsCollection]):
    control_keys = ["accuracy"]
    for name in ["ece", "brier", "auroc", "auprc"]:
        control_keys.extend([f"{name}_logits", f"{name}_verbalised"])

    final_dataframe = None
    for collection in collections:
        table = collection.generate_tables("Calibrator", control_keys)
        table["loss_fn"] = [collection.details["Loss Function"]] * len(table)

        if final_dataframe is None:
            final_dataframe = table
            continue
        final_dataframe = merge_dfs(final_dataframe, table)

    out_dict = collections[0].details.copy()
    del out_dict["Loss Function"]
    return final_dataframe, out_dict


def vary_calibrator_id(model_name: str, loss_func_name: str, prompt_version: PromptVersion, input_formatter: str):
    calib_collection = ModelMetricsCollection()
    calib_collection.details = {
        "Split": "Validation",
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

    id_if: Optional[InputFormatter] = None
    for calibrator_name in calibrator_names:
        print(sc.green(calibrator_name))
        calibrator_type = calibrator_dict[calibrator_name]
        if id_if is None:
            id_if = input_formatter_dict[input_formatter](llm_bundle,
                                                          prompt_version,
                                                          calibrator_type,
                                                          loss_func)
        else:
            # Ensure that the whole dataset doesn't end up being reloaded again.
            id_if.calibrator_type = calibrator_type

        # run the pipeline to ensure that all the calib and test results have been acquired.
        calib_data, test_data = id_if.run_pipeline(batch_size=4)

        details = {"Calibrator": calibrator_name}
        calib_results = ModelMetrics(calib_data, **details)
        test_results = ModelMetrics(test_data, **details)

        calib_collection.append(calib_results)
        test_collection.append(test_results)
    """
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
    """
    return calib_collection, test_collection


def main(model_name: str="google/gemma-2-2b-it",
         calibrator_name: str=None,
         loss_func_name: Optional[str]=None, #"CORRECT_AWARE",
         prompt_version: str="DEFAULT",
         id_input_formatter_name: str="SQUADV2CoT",
         ood_input_formatter_name: Optional[str]=None):
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
    if ood_input_formatter_name is None and calibrator_name is None and loss_func_name is None:
        calib_collections = []
        test_collections = []
        for lfn in ["BCE", "CORRECT_AWARE", "WEIGHTED_CORRECT_AWARE"]:
            print(sc.blue(lfn))
            calib_collection, test_collection = vary_calibrator_id(model_name,
                                                                   lfn,
                                                                   prompt_version,
                                                                   id_input_formatter_name)
            calib_collections.append(calib_collection)
            test_collections.append(test_collection)

        calib_compiled_df, calib_details = compare_collections_by_loss(calib_collections)
        test_compiled_df, test_details = compare_collections_by_loss(test_collections)

        print(tabulate(calib_details.items(), tablefmt="github"))
        print()
        print(calib_compiled_df.sort_values(by="ece_calib", ascending=True))
        print()
        print()
        print(tabulate(test_details.items(), tablefmt="github"))
        print()
        print(test_compiled_df.sort_values(by="ece_calib", ascending=True))

    elif ood_input_formatter_name is None and calibrator_name is None:
        print("vary_calibrator_id")
        vary_calibrator_id(model_name, loss_func_name, prompt_version, id_input_formatter_name)
    #elif ood_input_formatter_name is None:
    #    print("vary_ood_if")
    #    vary_ood_if(model_name, calibrator_name, prompt_version, id_input_formatter_name)
    elif calibrator_name is None and loss_func_name is None:
        collections = []
        for lfn in ["BCE", "CORRECT_AWARE", "WEIGHTED_CORRECT_AWARE"]:
            print(sc.blue(lfn))
            collections.append(vary_calibrator_ood(model_name,
                                                   prompt_version,
                                                   lfn,
                                                   id_input_formatter_name,
                                                   ood_input_formatter_name))
        compiled_df, details = compare_collections_by_loss(collections)

        print(tabulate(details.items(), tablefmt="github"))
        print()
        print(compiled_df.sort_values(by="ece_calib", ascending=True))

    elif calibrator_name is None:
        print("vary_calibrator_ood")
        vary_calibrator_ood(model_name, prompt_version, loss_func_name, id_input_formatter_name, ood_input_formatter_name)


if __name__ == "__main__":
    fire.Fire(main)
