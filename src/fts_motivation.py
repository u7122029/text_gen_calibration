import fire
import numpy as np
import torch
from pathlib import Path
import pathlib
import pandas as pd
from sklearn.linear_model import LinearRegression

from calibrators.frequency_ts import compute_top_bot_dfs, std_proc
from llm_models import TextGenLLMBundle
from data import DictDataset
from utils import RESULTS_PATH, FIGURES_PATH
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('text', usetex=True)


def compute_gamma(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)

    # Create matrices of pairwise differences
    x_diff = x[:, np.newaxis] - x[np.newaxis, :]
    y_diff = y[:, np.newaxis] - y[np.newaxis, :]

    # Calculate concordant and discordant pairs
    concordant = np.sum((x_diff * y_diff > 0) & (np.triu(np.ones((n, n)), k=1) > 0))
    discordant = np.sum((x_diff * y_diff < 0) & (np.triu(np.ones((n, n)), k=1) > 0))

    return (concordant - discordant) / (concordant + discordant)


def m_metric(mean, std, rfr):
    return mean


def s_metric(mean, std, response_frequency_ratio):
    sf_std = std_proc(std)
    return sf_std


def r_metric(mean, std, rfr):
    return rfr


def ms_metric(mean, std, rfr):
    sf_std = std_proc(std)
    return mean * sf_std


def sr_metric(mean, std, response_frequency_ratio):
    sf_std = std_proc(std)
    return sf_std * response_frequency_ratio


def mr_metric(mean, std, rfr):
    return mean * rfr


def msr_metric(mean, std, rfr):
    return mean * std_proc(std) * rfr


def zeroing_results(input_formatter_name, model_name):
    path = Path(RESULTS_PATH) / model_name / input_formatter_name / "DEFAULT" / "calib_data" / "data.dill"
    llm_bundle = TextGenLLMBundle(model_name)

    # First get calibration dset
    dset = DictDataset.from_file(path)
    top_df, bot_df = compute_top_bot_dfs(dset, llm_bundle, metric_func=sr_metric)

    high_xi_tokens = torch.Tensor(top_df[top_df["token_values"] >= 0.8]["token_ids"].to_numpy())
    low_xi_tokens = torch.Tensor(top_df[top_df["token_values"] <= 0.2]["token_ids"].to_numpy())
    dset_confs = dset["logits_confs"]
    adjust_high_xi0 = []
    adjust_low_xi0 = []

    adjust_high_xi1 = []
    adjust_low_xi1 = []

    llm_bundle.lm_head.cuda()
    for x in dset:
        logits = llm_bundle.final_hs_to_logits(x["final_hidden_states"]).cpu()
        tokens = x["tokens"]
        token_confs = torch.take_along_dim(torch.softmax(logits, dim=1), tokens.unsqueeze(1), dim=1).squeeze()

        mask_high = torch.isin(tokens, high_xi_tokens)
        mask_low = torch.isin(tokens, low_xi_tokens)

        token_confs_high0 = token_confs.clone()
        token_confs_high0[mask_high] = 0

        token_confs_high1 = token_confs.clone()
        token_confs_high1[mask_high] = 1

        token_confs_low0 = token_confs.clone()
        token_confs_low0[mask_low] = 0

        token_confs_low1 = token_confs.clone()
        token_confs_low1[mask_low] = 1

        adjust_high_xi0.append(torch.mean(token_confs_high0))
        adjust_low_xi0.append(torch.mean(token_confs_low0))

        adjust_high_xi1.append(torch.mean(token_confs_high1))
        adjust_low_xi1.append(torch.mean(token_confs_low1))

    adjust_low_xi0 = torch.Tensor(adjust_low_xi0)
    adjust_high_xi0 = torch.Tensor(adjust_high_xi0)

    adjust_low_xi1 = torch.Tensor(adjust_low_xi1)
    adjust_high_xi1 = torch.Tensor(adjust_high_xi1)

    plt.figure()
    plt.boxplot([dset_confs, adjust_high_xi0, adjust_low_xi0, adjust_high_xi1, adjust_low_xi1])
    plt.xticks([1, 2, 3, 4, 5], ['Control', r'$\geq M$, 0', r'$\leq 1 - M$, 0', r'$\geq M$, 1', r'$\leq 1 - M$, 1'])
    plt.title(rf"Response Confidence Distributions based on $\xi$-score ({model_name}).")
    plt.ylabel("Response Confidence")
    plt.show()
    #path = Path(FIGURES_PATH) / model_name
    #path.mkdir(parents=True, exist_ok=True)
    #plt.savefig(path / "zeroing.png", dpi=600, transparent=True)


def show_xi_scores(input_formatter_name, llm_bundle: TextGenLLMBundle, metric):
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    path = Path(RESULTS_PATH) / llm_bundle.llm_name / input_formatter_name / "DEFAULT" / "calib_data" / "data.dill"

    # First get calibration dset
    dset = DictDataset.from_file(path)
    top_df, bot_df = compute_top_bot_dfs(dset, llm_bundle, metric_func=metric)

    """lr = LinearRegression()
    lr.fit(top_df["stds_proc"].to_numpy().reshape(-1,1), top_df["means"])
    coef = lr.coef_
    y_int = lr.intercept_
    approx = lambda x: coef * x + y_int
    gamma = compute_gamma(top_df["stds_proc"], top_df["means"])
    x_low = top_df["stds_proc"].min()
    x_high = top_df["stds_proc"].max()
    plt.figure()
    plt.scatter(top_df["stds_proc"], top_df["means"])
    plt.plot([x_low, x_high], [approx(x_low), approx(x_high)],
             label=rf"$\gamma = {np.round(gamma, 4)}$",
             color="orange")
    plt.title(rf"$\mu$ vs. $s$, {input_formatter_name[:-3]}")
    plt.xlabel(r"$s$")
    plt.ylabel(r"$\mu$")
    plt.legend(loc="best")
    plt.tight_layout()

    pathlib.PosixPath = temp
    plt.show()"""
    print(top_df)
    pathlib.PosixPath = temp


def main(input_formatter_name: str="SQUADV2CoT",
         model_name="microsoft/Phi-3-mini-4k-instruct"):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)
    torch.manual_seed(0)

    zeroing_results(input_formatter_name, model_name)
    """llm_bundle = TextGenLLMBundle(model_name)
    metrics = [sr_metric]#, mr_metric, sr_metric, msr_metric]
    for metric in metrics:
        print(metric.__name__)
        show_xi_scores(input_formatter_name, llm_bundle, metric)"""


if __name__ == "__main__":
    fire.Fire(main)
