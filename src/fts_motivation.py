import fire
import torch
from pathlib import Path
import pandas as pd

from calibrators.frequency_ts import compute_top_bot_dfs, std_proc
from llm_models import TextGenLLMBundle
from input_formatters import input_formatter_dict
from prompt_formatters import PromptVersion
from data import DictDataset
from utils import RESULTS_PATH
import matplotlib.pyplot as plt


def fts_metric(mean, std, response_frequency_ratio):
    sf_std = std_proc(std)
    return mean * sf_std * response_frequency_ratio


def main(input_formatter_name: str="GSMCoT",
         model_name="google/gemma-1.1-2b-it"):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)
    torch.manual_seed(0)
    path = Path(RESULTS_PATH) / model_name / input_formatter_name / "CoTPromptFormat" / "calib_data" / "data.dill"
    llm_bundle = TextGenLLMBundle(model_name)
    # First get calibration dset
    dset = DictDataset.from_file(path)
    top_df, bot_df = compute_top_bot_dfs(dset, llm_bundle, metric_func=fts_metric)
    print(len(top_df))
    print(top_df)
    print()
    print(len(bot_df))
    print(bot_df)
    plt.figure()
    plt.scatter(top_df["stds_proc"], top_df["means"])
    plt.scatter(top_df["response_props"], top_df["means"])
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)