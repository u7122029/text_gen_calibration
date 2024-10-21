import dill
import torch
from transformers import AutoTokenizer
import transformers
print(transformers.__version__)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
dataset_name = "SQUADV2CoT"
ood_dataset_name = None
tokeniser = AutoTokenizer.from_pretrained(model_name)

n = 0
d = str(n).zfill(4)
print(d)

if ood_dataset_name is not None:
    with open(f"results/{model_name}/{ood_dataset_name}/DEFAULT/test_data/data.dill", "rb") as f:
        x = dill.load(f)
        print("keys:", x.keys())
        print(x["question"][n])
        print(x["correct"][n])
        print(x["answer"][n])
        print(x["logits_confs"][n])
        print()

    with open(f"results/{model_name}/{ood_dataset_name}/DEFAULT/test_data/{d}/tokens.dill", "rb") as f:
        x = dill.load(f)
        print(tokeniser.decode(x))

    with open(f"results/{model_name}/{ood_dataset_name}/DEFAULT/test_data/{d}/token_probs.dill", "rb") as f:
        x = dill.load(f)
        print(x.mean())

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/CORRECT_AWARE/APRICOT_FLHS_MR/ood/DEFAULT/{ood_dataset_name}.dill", "rb") as f:
        x = dill.load(f)
        print(x["calibrated_confs"][n])

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/CORRECT_AWARE/APRICOT_LHS/ood/DEFAULT/{ood_dataset_name}.dill", "rb") as f:
        x = dill.load(f)
        print(x["calibrated_confs"][n])
else:
    with open(f"results/{model_name}/{dataset_name}/DEFAULT/test_data/data.dill", "rb") as f:
        x = dill.load(f)
        print("keys:", x.keys())
        print(x["context"][n])
        print()
        print(x["question"][n])
        print(x["correct"][n])
        print(x["answer"][n])
        print(x["logits_confs"][n])
        print(x["numeric_confs"][n])
        print(x["worded_confs"][n])
        print()

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/test_data/{d}/tokens.dill", "rb") as f:
        x = dill.load(f)
        token_list = tokeniser.batch_decode(x)

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/test_data/{d}/token_probs.dill", "rb") as f:
        x = dill.load(f)
        original_token_probs = x

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/CORRECT_AWARE/TemperatureScaling/test_results.dill", "rb") as f:
        x = dill.load(f)
        print(x.keys())
        print(x["calibrated_confs"][n])
        token_confs = x["calibrated_token_probs"][n]
    print(token_confs)

    print(len(token_list))
    print(len(token_confs))
    for t, c in zip(token_list, original_token_probs):
        print(f"\\confnew{{{round(c.item(), 2)}}}{{{t}}}{{5}}")
