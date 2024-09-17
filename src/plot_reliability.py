import fire
from pathlib import Path
from utils import dill_load, RESULTS_PATH, FIGURES_PATH
from data import DictDataset
import torch


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib import colormaps
import matplotlib.patheffects as path_effects


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
            color="blue",
            label="Perfect Calibration")

    # Plot histogram bars with error
    bar_width = 1 / n_bins
    bar_centers = bin_edges[:-1] + bar_width / 2

    legend_set = False
    for bar_centre, prob, total in zip(bar_centers, prob_true, bin_total):
        if total == 0:
            continue

        kwargs = {}
        if not legend_set:
            kwargs["label"] = "Error"
            legend_set = True

        ax.plot([bar_centre, bar_centre], [prob, bar_centre], linewidth=2, color="green", **kwargs)

    # Create a color map and normalize bin_total values
    cmap = colormaps["Wistia"]
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

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Samples in bin')

    return fig, ax


def main(model_name="microsoft/Phi-3-mini-4k-instruct",
         input_formatter="SQUADV2CoT",
         loss_type="CORRECT_AWARE",
         prompt_formatter="WordAnswerCoTPromptFormat",
         calibrator_name="APRICOT_Original"):

    data_path = Path(RESULTS_PATH) / model_name / input_formatter / prompt_formatter
    figures_path = Path(FIGURES_PATH) / model_name / input_formatter / prompt_formatter / loss_type / calibrator_name
    test_data = DictDataset.from_file(data_path / "test_data" / "data.dill")
    test_results = dill_load(data_path / loss_type / calibrator_name / "test_results.dill")
    test_data = test_data.update(test_results)
    print(test_data.keys())

    # Plot and save figures
    fig, ax = reliability_diagram(test_data["correct"], test_data["logits_confs"], f"Logit-based Confidences ({model_name})")
    fig.savefig(figures_path / "logits.png", dpi=600)
    fig1, ax1 = reliability_diagram(test_data["correct"], test_data["worded_confs"], f"Verbalised Confidences ({model_name})")
    fig1.savefig(figures_path / "verbalised.png", dpi=600)
    fig2, ax2 = reliability_diagram(test_data["correct"], test_data["calibrated_confs"],
                                    f"Calibrated Confidences ({model_name}, {calibrator_name})")
    fig2.savefig(figures_path / "calibrated.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)