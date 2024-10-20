import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

from llm_models import TextGenLLMBundle

np.random.seed(1)


def make_ts_figure(z, temp, fname):
    p = sp.special.softmax(z/temp)
    print(p)
    plt.figure()
    plt.bar(np.arange(6) + 1, p)
    plt.ylabel("Probability")
    plt.xlabel("Logit Index")
    plt.title(fr"Probability Dist. ($\tau={round(temp, 4)}$)")
    plt.ylim([0,1])
    plt.tight_layout()
    plt.savefig(fname, dpi=300)


def make_sigmoid_figure():
    xs = np.linspace(-5,5, 1000)
    ys = sp.special.expit(xs)

    plt.figure()

    plt.plot(xs, ys)

    plt.xlim(-5, 5)
    plt.ylim(0, 1)

    plt.gca().spines['left'].set_position(('data', 0))
    plt.gca().spines['right'].set_visible(False)

    plt.tick_params(axis='x', direction='in', pad=10)
    plt.tick_params(axis='y', direction='in', pad=10)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

    plt.show()


def show_llm_sizes():
    name_dict = {
        # confirmed models.
        "google/gemma-2-2b-it",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        #"Zyphra/Zamba2-2.7B-instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B-Instruct"
    }

    out_dict = {
        "LLM": [],
        "#Params": [],
        "Hidden State Size": [],
        "Vocab Size": []
    }
    for name in name_dict:
        llm_bundle = TextGenLLMBundle(name)
        llm_bundle.load_model()
        out_dict["LLM"].append(llm_bundle.llm_name)
        out_dict["#Params"].append(sum(p.numel() for p in llm_bundle.llm_model.parameters()))
        out_dict["Hidden State Size"].append(llm_bundle.hidden_features)
        out_dict["Vocab Size"].append(llm_bundle.vocab_size())
        llm_bundle.unload_model()
        del llm_bundle

    out_dict = pd.DataFrame(out_dict).sort_values("#Params")
    print(out_dict)


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    show_llm_sizes()
    """make_sigmoid_figure()
    plt.show()"""

    """
    z = (np.random.random(6) * 2 - 1) * 2
    make_figure(z, 1, "ts1.png")
    make_figure(z, 3, "ts2.png")
    make_figure(z, 1/3, "ts3.png")"""