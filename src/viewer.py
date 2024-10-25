import dill
import torch
from transformers import AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
dataset_name = "SQUADV2CoT"
ood_dataset_name = None
tokeniser = AutoTokenizer.from_pretrained(model_name)
loss = "WEIGHTED_CORRECT_AWARE"
calibrator = "FrequencyTS_MR"

n = 49 # 40 is the default for ID!!!
d = str(n).zfill(4)
#print(d)

if ood_dataset_name is not None:
    with open(f"results/{model_name}/{ood_dataset_name}/DEFAULT/test_data/data.dill", "rb") as f:
        x = dill.load(f)
        print("keys:", x.keys())
        print(x["question"][n])
        print(x["correct"][n])
        print("ANSWER:",x["answer"][n])
        print("LOGIT_CONF:",x["logits_confs"][n])
        print("NUM_CONF:", x["numeric_confs"][n])
        print("WORD_CONF:", x["worded_confs"][n])
        print()

    with open(f"results/{model_name}/{ood_dataset_name}/DEFAULT/test_data/{d}/tokens.dill", "rb") as f:
        x = dill.load(f)
        print(tokeniser.decode(x))

    #with open(f"results/{model_name}/{ood_dataset_name}/DEFAULT/test_data/{d}/token_probs.dill", "rb") as f:
        #x = dill.load(f)
        #print(x.mean())

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/CORRECT_AWARE/{calibrator}/ood/DEFAULT/{ood_dataset_name}.dill", "rb") as f:
        x = dill.load(f)
        print(x["calibrated_confs"][n])
else:
    with open(f"results/{model_name}/{dataset_name}/DEFAULT/test_data/{d}/tokens.dill", "rb") as f:
        x = dill.load(f)
        token_list = tokeniser.batch_decode(x)

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/test_data/data.dill", "rb") as f:
        x = dill.load(f)
        correct = x["correct"][n]
        print("correct:", correct)

        #print("keys:", x.keys())
        try:
           print(x["context"][n])
        except:
            pass

        print()
        print(x["question"][n])

        print("ANSWER(S):",x["answer"][n])
        print("LOGIT CONFS:",x["logits_confs"][n])
        print("NUMERIC CONF:",x["numeric_confs"][n])
        print("WORDED CONF:",x["worded_confs"][n])
        print()

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/test_data/{d}/token_probs.dill", "rb") as f:
        x = dill.load(f)
        original_token_probs = x

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/{loss}/{calibrator}/test_results.dill", "rb") as f:
        x = dill.load(f)
        #print(x.keys())
        print("CALIBRATED CONF:",x["calibrated_confs"][n])
        token_confs = x["calibrated_token_probs"][n]
    #print(token_confs)

    print(len(token_list))
    print(len(token_confs))
    for t, c in zip(token_list, token_confs):
        print(f"\\confnew{{{round(c.item(), 2)}}}{{{t}}}{{5}}")

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/{loss}/{calibrator}/calib_weights.dill",
              "rb") as f:
        x = dill.load(f)
        print(x)
