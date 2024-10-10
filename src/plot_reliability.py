from typing import Optional

import fire
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, Normalize

from calibrators import calibrator_dict, compute_top_bot_dfs, std_proc
from input_formatters import input_formatter_dict
from llm_models import TextGenLLMBundle
from prompt_formatters import PromptVersion
from utils import dill_load, RESULTS_PATH, FIGURES_PATH, LossFunc
from metrics import ModelMetrics

from torchmetrics.classification import BinaryCalibrationError
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


def boxplots(confs_uncalibrated, confs_xi, confs_non_xi):
    fig, ax = plt.subplots(figsize=(17, 8.1))
    ax.boxplot([confs_uncalibrated, confs_xi, confs_non_xi], vert=False)
    ax.set_yticklabels(['Original', 'xi > 0.8', 'xi <= 0.8'], fontsize=20)
    ax.set_xlabel('Confidence', fontsize=20)
    ax.set_title('Word Confidence Distributions', fontsize=20)
    ax.set_xlim(0.5, 1)
    fig.tight_layout()
    return fig, ax


def reliability_diagram(preds, confs, title, n_bins=15):
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()

    if isinstance(confs, torch.Tensor):
        confs = confs.numpy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confs, bin_edges[1: -1])

    bin_sums = np.bincount(bin_indices, weights=confs, minlength=n_bins)
    bin_correct = np.bincount(bin_indices, weights=preds, minlength=n_bins)
    bin_total = np.bincount(bin_indices, minlength=n_bins)

    prob_true = np.zeros(n_bins)
    np.divide(bin_correct, bin_total, out=prob_true, where=bin_total != 0)

    prob_pred = np.zeros(n_bins)
    np.divide(bin_sums, bin_total, out=prob_pred, where=bin_total != 0)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the diagonal line
    ax.plot([0, 1],
            [0, 1],
            linestyle="--",
            color="green",
            label="Perfect Alignment")

    # Plot histogram bars with error
    bar_width = 1 / n_bins
    bar_centers = bin_edges[:-1] + bar_width / 2

    colors = [(1, 0.7, 0.7), (0.6, 0, 0)]  # light red to dark red

    # Create the colormap
    cmap_error = LinearSegmentedColormap.from_list("red_uniform", colors, N=20)

    norm_error = Normalize(vmin=0, vmax=np.sum(bin_total))
    largest_bin = np.max(bin_total)
    for bar_centre, prob, total in zip(bar_centers, prob_true, bin_total):
        if total == 0:
            continue

        kwargs = {}
        if total == largest_bin:
            kwargs["label"] = "Misalignment"

        ax.plot([bar_centre, bar_centre], [prob, bar_centre],
                linewidth=2,
                color=cmap_error(norm_error(total)),
                **kwargs)

    # Create a color map and normalize bin_total values
    colors = ['#FFFFFF', '#E6F2FF', '#CCE5FF', '#99CCFF', '#66B2FF', '#3399FF', '#0080FF', '#0066CC', '#004C99',
              '#003366', '#001A33']

    # Create the colormap
    cmap = LinearSegmentedColormap.from_list("dense_map", colors, N=20)
    #cmap = colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=np.sum(bin_total))
    bars = ax.bar(bar_centers, prob_true, width=bar_width, alpha=0.7, edgecolor='black',
                  color=[cmap(norm(count)) for count in bin_total])

    # Add counts on top of bars
    for rect, count in zip(bars, bin_total):
        if count == 0:
            continue
        height = rect.get_height()
        text = ax.text(rect.get_x() + rect.get_width() / 2., height - 0.035,
                       f'{count}', color="white", ha='center', va='bottom')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    ax.set_xlabel("Confidence", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(prop={'size': 24})
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()

    # Add a colorbar
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #cbar = plt.colorbar(sm, ax=ax, label='Samples in bin')

    return fig, ax


def plot_ood(model_name: str,
             calibrator_name: str,
             #id_prompt_version: str,
             id_prompt_version: str,
             ood_prompt_version: str,
             loss_func_name: str,
             id_if_name: str,
             ood_if_name: str):
    id_prompt_version = PromptVersion.from_string(id_prompt_version)
    ood_prompt_version = PromptVersion.from_string(ood_prompt_version)

    figures_path = (Path(FIGURES_PATH) /
                    model_name /
                    id_if_name /
                    id_prompt_version.name /
                    loss_func_name /
                    calibrator_name /
                    ood_if_name /
                    ood_prompt_version.name)
    figures_path.mkdir(parents=True, exist_ok=True)

    llm_bundle = TextGenLLMBundle(model_name)
    loss_func = LossFunc.from_string(loss_func_name)

    calibrator_type = calibrator_dict[calibrator_name]
    id_if = input_formatter_dict[id_if_name](llm_bundle,
                                             id_prompt_version,
                                             calibrator_type,
                                             loss_func)

    ood_if = input_formatter_dict[ood_if_name](llm_bundle,
                                               ood_prompt_version,
                                               calibrator_type,
                                               loss_func)

    test_data = ood_if.test_calibrator(id_if)
    processed = ModelMetrics(test_data)

    # Plot and save figures
    fig, ax = reliability_diagram(processed.correct, processed.logits_confs,
                                  f"Logit-based Responses ({model_name})")
    fig1, ax1 = reliability_diagram(processed.correct, processed.verbalised_confs,
                                    f"Verbalised Responses ({model_name})")
    fig2, ax2 = reliability_diagram(processed.correct, processed.calibrated_confs,
                                    f"Calibrated Responses ({calibrator_name})")
    fig2.savefig(figures_path / "calibrated.png", dpi=600)

    print(f"ECE: {processed.ece_calibrated}")

    # First get highest tokens by threshold
    def metric(mean, std, response_frequency_ratio):
        return response_frequency_ratio

    df_top, bot_df = compute_top_bot_dfs(test_data, llm_bundle, metric)
    df_top = df_top[df_top["token_values"] >= 0.8]
    token_ids = torch.Tensor(df_top["token_ids"].to_numpy()).int()

    # Plot token-based reliability diagrams

    modified_confs = []
    modified_confs1 = []

    calibrated_token_confs = []
    calibrated_token_confs1 = []
    for outcome, token_confs, tokens, calib_token_confs, successful in zip(test_data["correct"],
                                                                           test_data["token_probs"],
                                                                           test_data["tokens"],
                                                                           test_data["calibrated_token_probs"],
                                                                           processed.logit_confs_successful):
        if not successful: continue

        mask = torch.isin(tokens, token_ids)
        modified_confs.append(token_confs[mask].mean())
        modified_confs1.append(token_confs[~mask].mean())

        calibrated_token_confs.append(calib_token_confs[mask].mean())
        calibrated_token_confs1.append(calib_token_confs[~mask].mean())

    modified_confs = torch.Tensor(modified_confs)
    modified_confs1 = torch.Tensor(modified_confs1)

    calibrated_token_confs = torch.Tensor(calibrated_token_confs)
    calibrated_token_confs1 = torch.Tensor(calibrated_token_confs1)

    fig3, ax3 = reliability_diagram(processed.correct,
                                    modified_confs,
                                    f"xi > 0.8 Responses", n_bins=30)
    fig4, ax4 = reliability_diagram(processed.correct,
                                    modified_confs1,
                                    f"xi <= 0.8 Responses", n_bins=30)

    fig5, ax5 = reliability_diagram(processed.correct,
                                    calibrated_token_confs,
                                    f"xi > 0.8 Responses (calib)", n_bins=30)
    fig6, ax6 = reliability_diagram(processed.correct,
                                    calibrated_token_confs1,
                                    f"xi <= 0.8 Responses (calib)", n_bins=30)

    fig5.savefig(figures_path / "xi.png", dpi=600)
    fig6.savefig(figures_path / "non_xi.png", dpi=600)

    fig7, ax7 = boxplots(processed.logits_confs, modified_confs, modified_confs1)
    plt.show()


def plot_id(model_name="microsoft/Phi-3-mini-4k-instruct",
            input_formatter_name="SQUADV2CoT",
            loss_func_name="CORRECT_AWARE",
            prompt_version="DEFAULT",
            calibrator_name="APRICOT_TemperatureScaling"):
    prompt_version = PromptVersion.from_string(prompt_version)
    figures_path = (Path(FIGURES_PATH) /
                    model_name /
                    input_formatter_name /
                    prompt_version.name /
                    loss_func_name /
                    calibrator_name)
    figures_path.mkdir(parents=True, exist_ok=True)

    llm_bundle = TextGenLLMBundle(model_name)
    loss_func = LossFunc.from_string(loss_func_name)

    calibrator_type = calibrator_dict[calibrator_name]
    input_formatter = input_formatter_dict[input_formatter_name](llm_bundle, prompt_version, calibrator_type, loss_func)
    _, test_data = input_formatter.run_pipeline()
    print(test_data.keys())

    # Plot response-based reliability diagrams
    fig, ax = reliability_diagram(test_data["correct"], test_data["logits_confs"],
                                  f"Logit-based Responses ({model_name})")
    fig.savefig(figures_path.parent.parent / "logits.png", dpi=600)

    fig1, ax1 = reliability_diagram(test_data["correct"], test_data["worded_confs"],
                                    f"Verbalised Responses ({model_name})")
    fig1.savefig(figures_path.parent.parent / "verbalised.png", dpi=600)

    fig2, ax2 = reliability_diagram(test_data["correct"], test_data["calibrated_confs"],
                                    f"Calibrated Responses ({calibrator_name})")
    fig2.savefig(figures_path / "calibrated.png", dpi=600)

    # First get highest tokens by threshold
    def metric(mean, std, response_frequency_ratio):
        return response_frequency_ratio

    df_top, bot_df = compute_top_bot_dfs(test_data, llm_bundle, metric)
    df_top = df_top[df_top["token_values"] >= 0.8]
    token_ids = torch.Tensor(df_top["token_ids"].to_numpy()).int()

    # Plot token-based reliability diagrams

    modified_confs = []
    modified_confs1 = []

    calibrated_token_confs = []
    calibrated_token_confs1 = []
    for outcome, token_confs, tokens, calib_token_confs in zip(test_data["correct"], test_data["token_probs"], test_data["tokens"], test_data["calibrated_token_probs"]):
        mask = torch.isin(tokens, token_ids)
        modified_confs.append(token_confs[mask].mean())
        modified_confs1.append(token_confs[~mask].mean())

        calibrated_token_confs.append(calib_token_confs[mask].mean())
        calibrated_token_confs1.append(calib_token_confs[~mask].mean())

    modified_confs = torch.Tensor(modified_confs)
    modified_confs1 = torch.Tensor(modified_confs1)

    calibrated_token_confs = torch.Tensor(calibrated_token_confs)
    calibrated_token_confs1 = torch.Tensor(calibrated_token_confs1)

    fig3, ax3 = reliability_diagram(test_data["correct"], modified_confs,
                                    f"xi > 0.8 Responses", n_bins=30)
    fig4, ax4 = reliability_diagram(test_data["correct"], modified_confs1,
                                    f"xi <= 0.8 Responses", n_bins=30)

    fig5, ax5 = reliability_diagram(test_data["correct"], calibrated_token_confs,
                                    f"xi > 0.8 Responses (calibrated)", n_bins=30)
    fig6, ax6 = reliability_diagram(test_data["correct"], calibrated_token_confs1,
                                    f"x <= 0.8 Responses (calibrated)", n_bins=30)

    fig3.savefig(figures_path.parent.parent / "xi.png", dpi=600)
    fig4.savefig(figures_path.parent.parent / "non_xi.png", dpi=600)
    fig5.savefig(figures_path / "calib_xi.png", dpi=600)
    fig6.savefig(figures_path / "calib_non_xi.png", dpi=600)

    fig7, ax7 = boxplots(test_data["logits_confs"], modified_confs, modified_confs1)
    fig7.savefig(figures_path.parent.parent / "box_comparisons.png", dpi=600)
    mm = ModelMetrics(test_data)
    print(f"ECE: {mm.ece_calibrated}")
    plt.show()


def main(model_name: str = "Zyphra/Zamba2-2.7B-instruct",
         calibrator_name: str = "APRICOT_FLHS_M",
         id_prompt_version: str = "DEFAULT",
         ood_prompt_version: str = "DEFAULT",
         loss_func_name: str = "CORRECT_AWARE",
         id_if_name: str = "SQUADV2CoT",
         ood_if_name: Optional[str] = "GSMCoT"):
    if ood_if_name is None:
        plot_id(model_name, id_if_name, loss_func_name, id_prompt_version, calibrator_name)
    else:
        plot_ood(model_name, calibrator_name, id_prompt_version, ood_prompt_version, loss_func_name, id_if_name, ood_if_name)


if __name__ == "__main__":
    fire.Fire(main)
