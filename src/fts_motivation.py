import fire
import torch
from pathlib import Path
import pandas as pd

from calibrators.frequency_ts import compute_top_bot_dfs, std_proc
from llm_models import TextGenLLMBundle
from data import DictDataset
from utils import RESULTS_PATH, FIGURES_PATH
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('text', usetex=True)


def fts_metric(mean, std, response_frequency_ratio):
    sf_std = std_proc(std)
    return sf_std * response_frequency_ratio


def zeroing_results(input_formatter_name, model_name):
    path = Path(RESULTS_PATH) / model_name / input_formatter_name / "CoTPromptFormat" / "val_data" / "data.dill"
    llm_bundle = TextGenLLMBundle(model_name)

    # First get calibration dset
    dset = DictDataset.from_file(path)
    top_df, bot_df = compute_top_bot_dfs(dset, llm_bundle, metric_func=fts_metric)

    high_xi_tokens = torch.Tensor(top_df[:5]["token_ids"].to_numpy())
    low_xi_tokens = torch.Tensor(top_df[-5:]["token_ids"].to_numpy())
    dset_confs = dset["logits_confs"]
    adjust_high_xi = []
    adjust_low_xi = []

    llm_bundle.lm_head.cuda()
    for x in dset:
        logits = llm_bundle.final_hs_to_logits(x["final_hidden_states"].cuda()).cpu()
        tokens = x["tokens"]
        token_confs = torch.take_along_dim(torch.softmax(logits, dim=1), tokens.unsqueeze(1), dim=1).squeeze()

        mask_high = torch.isin(tokens, high_xi_tokens)
        mask_low = torch.isin(tokens, low_xi_tokens)

        token_confs_high = token_confs.clone()
        token_confs_high[mask_high] = 0

        token_confs_low = token_confs.clone()
        token_confs_low[mask_low] = 0
        adjust_high_xi.append(torch.mean(token_confs_high))
        adjust_low_xi.append(torch.mean(token_confs_low))

    adjust_low_xi = torch.Tensor(adjust_low_xi)
    adjust_high_xi = torch.Tensor(adjust_high_xi)

    plt.figure()
    plt.boxplot([dset_confs, adjust_high_xi, adjust_low_xi])
    plt.xticks([1, 2, 3], ['Original', 'Top 5 Removed', 'Bottom 5 Removed'])
    plt.title(rf"Response Confidence Distributions based on $\xi$-score ({model_name}).")
    plt.ylabel("Response Confidence")

    path = Path(FIGURES_PATH) / model_name
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / "zeroing.png", dpi=600, transparent=True)


def show_xi_scores(input_formatter_name, model_name):
    path = Path(RESULTS_PATH) / model_name / input_formatter_name / "CoTPromptFormat" / "val_data" / "data.dill"
    llm_bundle = TextGenLLMBundle(model_name)

    # First get calibration dset
    dset = DictDataset.from_file(path)
    top_df, bot_df = compute_top_bot_dfs(dset, llm_bundle, metric_func=fts_metric)

def main(input_formatter_name: str="GSMCoT",
         model_name="microsoft/Phi-3-mini-128k-instruct"):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)
    torch.manual_seed(0)

    zeroing_results(input_formatter_name, model_name)
    """
    print(len(top_df))
    print(top_df)
    print()
    print(len(bot_df))
    print(bot_df)
    plt.figure()
    plt.scatter(top_df["stds_proc"], top_df["means"])
    plt.scatter(top_df["response_props"], top_df["means"])
    plt.show()"""


if __name__ == "__main__":
    fire.Fire(main)
