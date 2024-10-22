from typing import Optional

import fire
import pandas as pd
import simple_colors as sc
import torch.cuda
from tabulate import tabulate

from calibrators import calibrator_dict
from input_formatters import input_formatter_dict, InputFormatter
from llm_models.textgen import TextGenLLMBundle
from metrics import ModelMetrics, ModelMetricsCollection
from prompt_formatters import PromptVersion
from utils import LossFunc

calibrator_names = [
    # "APRICOT_Original",
    # "TokenCalibrator",
    # "TemperatureScaling",
    # "APRICOT_TemperatureScaling",
    "FrequencyPTS_M",
    "FrequencyPTS_S",
    "FrequencyPTS_R",
    "FrequencyPTS_MR",
    "FrequencyPTS_SR",
    # "LastHiddenStateCalibrator",
    # "APRICOT_LHS",
    "FrequencyTS_M",
    "FrequencyTS_S",
    "FrequencyTS_R",
    "FrequencyTS_MR",
    "FrequencyTS_SR",
    "APRICOT_FrequencyTS_M",
    "APRICOT_FrequencyTS_S",
    "APRICOT_FrequencyTS_R",
    "APRICOT_FrequencyTS_SR",
    "APRICOT_FrequencyTS_MR",
    "FLHS_M",
    "FLHS_S",
    "FLHS_R",
    "FLHS_SR",
    "FLHS_MR",
    "APRICOT_FLHS_M",
    "APRICOT_FLHS_S",
    "APRICOT_FLHS_R",
    "APRICOT_FLHS_SR",
    "APRICOT_FLHS_MR",
    # "LogitConfsPlattScaling",
    # "FTP_M",
    # "FTP_S",
    # "FTP_R",
    # "FTP_MR",
    # "FTP_SR",
    # "APRICOT_FTP_M",
    # "APRICOT_FTP_S",
    # "APRICOT_FTP_R",
    # "APRICOT_FTP_MR",
    # "APRICOT_FTP_SR",
    # "FPS_M",
    # "FPS_S",
    # "FPS_R",
    # "FPS_MR",
    # "FPS_SR",
    # "APRICOT_FPS_M",
    # "APRICOT_FPS_S",
    # "APRICOT_FPS_R",
    # "APRICOT_FPS_MR",
    # "APRICOT_FPS_SR"
]

def vary_calibrator_ood(model_name: str,
                        id_prompt_version: PromptVersion,
                        ood_prompt_version: PromptVersion,
                        loss_func_name: str,
                        id_if_name: str,
                        ood_if_name: str):
    llm_bundle = TextGenLLMBundle(model_name)
    loss_func = LossFunc.from_string(loss_func_name)

    collection = ModelMetricsCollection()
    collection.details = {
        "LLM": model_name,
        "Prompt Version": id_prompt_version.name,
        "Loss Function": loss_func_name,
        "Calib. Input Formatter": id_if_name,
        "Test Input Formatter": ood_if_name
    }
    ood_if: Optional[InputFormatter] = None
    id_if: Optional[InputFormatter] = None
    for calibrator_name in calibrator_names:
        print(sc.green(calibrator_name))
        if calibrator_name.startswith("APRICOT") and loss_func.name.startswith("WEIGHTED"):
            print("APRICOT models have no class imbalance. Skipping.")
            continue

        calibrator_type = calibrator_dict[calibrator_name]
        if id_if is None:
            id_if = input_formatter_dict[id_if_name](llm_bundle, id_prompt_version, calibrator_type, loss_func)
        else:
            id_if.calibrator_type = calibrator_type

        if ood_if is None:
            ood_if = input_formatter_dict[ood_if_name](llm_bundle,
                                                       ood_prompt_version,
                                                       calibrator_type,
                                                       loss_func)
        else:
            ood_if.calibrator_type = calibrator_type

        test_results = ood_if.test_calibrator(id_if)

        details = {"Calibrator": calibrator_name}
        collection.append(ModelMetrics(test_results, **details))
        del test_results

    return collection


def merge_dfs(*dfs):
    df = pd.concat(*dfs).reset_index()
    #groups = df.groupby("Calibrator")
    return df.loc[df.groupby('Calibrator')['ece_calib'].idxmin()]


def compare_collections_by_loss(collections: list[ModelMetricsCollection]):
    control_keys = ["accuracy"]
    for name in ["ece", "brier", "auroc", "auprc"]:
        control_keys.extend([f"{name}_logits", f"{name}_verbalised"])

    dfs = []
    for collection in collections:
        table = collection.generate_tables("Calibrator", control_keys)
        table["loss_fn"] = [collection.details["Loss Function"]] * len(table)

        dfs.append(table)

    out_dict = collections[0].details.copy()
    del out_dict["Loss Function"]
    return merge_dfs(dfs), out_dict


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

    id_if: Optional[InputFormatter] = None
    for calibrator_name in calibrator_names:
        print(sc.green(calibrator_name))
        if calibrator_name.startswith("APRICOT") and loss_func.name.startswith("WEIGHTED"):
            print("APRICOT models have no class imbalance. Skipping.")
            continue
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
    return calib_collection, test_collection


def modify_calib_names(df: pd.DataFrame):
    def renaming(name: str):
        apricot = False
        if "APRICOT" in name:
            name = name.replace("APRICOT", "A")
            apricot = True

        match name:
            case "TokenCalibrator":
                return "RE"
            case "LogitConfsPlattScaling":
                return "PS"
            case "LastHiddenStateCalibrator":
                return "LHS"
            case "A_Original":
                return "A\\_RE"

        parts = name.split("_")
        if apricot:
            idx = 1
        else:
            idx = 0

        target = parts[idx]
        match target:
            case "FTP":
                modified = f"\\(\\xi_{{{parts[idx + 1].lower()}}}\\)-TP"
            case "FPS":
                modified = f"\\(\\xi_{{{parts[idx + 1].lower()}}}\\)-PS"
            case "FLHS":
                modified = f"\\(\\xi_{{{parts[idx + 1].lower()}}}\\)-LHS"
            case "FrequencyTS":
                modified = f"\\(\\xi_{{{parts[idx + 1].lower()}}}\\)-TS"
            case "FrequencyPTS":
                modified = f"\\(\\xi_{{{parts[idx + 1].lower()}}}\\)-PTS"
            case "TemperatureScaling":
                modified = "TS"
            case _:
                modified = target
        lst = []
        if apricot:
            lst = ["A"]
        lst.append(modified)
        return "\\_".join(lst)

    df["Calibrator"] = df["Calibrator"].apply(renaming)
    return df


def modify_loss_names(df: pd.DataFrame):
    def renaming(name: str):
        match name:
            case "WEIGHTED_BCE":
                name = "W-BCE\\\\"
            case "WEIGHTED_CORRECT_AWARE":
                name = "W-CA\\\\"
            case "CORRECT_AWARE":
                name = "CA\\\\"
            case _:
                name += "\\\\"
        return name
    df["loss_fn"] = df["loss_fn"].apply(renaming)
    return df


def main(model_name: str="google/gemma-2-2b-it",
         calibrator_name: str=None,
         loss_func_name: Optional[str]=None, #"CORRECT_AWARE",
         id_prompt_version: str="DEFAULT",
         ood_prompt_version: str="DEFAULT",
         id_input_formatter_name: str="SQUADV2CoT",
         ood_input_formatter_name: Optional[str]=None):
    """

    @param model_name:
    @param calibrator_name:
    @param loss_func_name:
    @param id_prompt_version:
    @param ood_prompt_version:
    @param id_input_formatter_name:
    @param ood_input_formatter_name:
    @return:
    """
    id_prompt_version = PromptVersion.from_string(id_prompt_version)
    ood_prompt_version = PromptVersion.from_string(ood_prompt_version)

    # Check that exactly one of the arguments is None.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)

    if ood_input_formatter_name is None and calibrator_name is None and loss_func_name is None:
        print("Comparing Loss Functions (ID).")
        calib_collections = []
        test_collections = []
        for lfn in ["BCE", "WEIGHTED_BCE", "CORRECT_AWARE", "WEIGHTED_CORRECT_AWARE"]:
            print(sc.blue(lfn))
            calib_collection, test_collection = vary_calibrator_id(model_name,
                                                                   lfn,
                                                                   id_prompt_version,
                                                                   id_input_formatter_name)
            calib_collections.append(calib_collection)
            test_collections.append(test_collection)

        calib_compiled_df, calib_details = compare_collections_by_loss(calib_collections)
        test_compiled_df, test_details = compare_collections_by_loss(test_collections)
        calib_compiled_df = modify_loss_names(modify_calib_names(calib_compiled_df))
        test_compiled_df = modify_loss_names(modify_calib_names(test_compiled_df))

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
        vary_calibrator_id(model_name, loss_func_name, id_prompt_version, id_input_formatter_name)
    #elif ood_input_formatter_name is None:
    #    print("vary_ood_if")
    #    vary_ood_if(model_name, calibrator_name, id_prompt_version, id_input_formatter_name)
    elif calibrator_name is None and loss_func_name is None:
        print("Comparing loss functions (OOD).")
        collections = []
        loss_names = [
            "BCE",
            "WEIGHTED_BCE",
            "CORRECT_AWARE",
            "WEIGHTED_CORRECT_AWARE"
        ]
        for lfn in loss_names:
            print(sc.blue(lfn))
            collections.append(vary_calibrator_ood(model_name,
                                                   id_prompt_version,
                                                   ood_prompt_version,
                                                   lfn,
                                                   id_input_formatter_name,
                                                   ood_input_formatter_name))
            torch.cuda.empty_cache()
        compiled_df, details = compare_collections_by_loss(collections)
        compiled_df = modify_loss_names(modify_calib_names(compiled_df))
        print(tabulate(details.items(), tablefmt="github"))
        print()
        print(compiled_df.sort_values(by="ece_calib", ascending=True))

    elif calibrator_name is None:
        print("vary_calibrator_ood")
        vary_calibrator_ood(model_name, id_prompt_version, ood_prompt_version, loss_func_name, id_input_formatter_name, ood_input_formatter_name)


if __name__ == "__main__":
    fire.Fire(main)
