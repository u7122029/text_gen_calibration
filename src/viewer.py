import dill
import torch
from transformers import AutoTokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
dataset_name = "SQUADV2CoT"
ood_dataset_name = None
tokeniser = AutoTokenizer.from_pretrained(model_name)
loss = "CORRECT_AWARE"
calibrator = "APRICOT_FLHS_SR"

n = 49 # 40 is the default for ID!!!
d = str(n).zfill(4)
#print(d)

if ood_dataset_name is not None:
    for i in range(100, 400):
        with open(f"results/{model_name}/{ood_dataset_name}/DEFAULT/test_data/data.dill", "rb") as f:
            x = dill.load(f)
            correct = x["correct"][i]
            if correct == 1:
                continue
            n = i
            d = str(n).zfill(4)
            #print("keys:", x.keys())
            print(f"\nn = {n}")
            print(x["question"][n])
            print(x["correct"][n])
            print("ANSWER:",x["answer"][n])
            print("LOGIT_CONF:",x["logits_confs"][n])
            print("NUM_CONF:", x["numeric_confs"][n])
            print("WORD_CONF:", x["worded_confs"][n])
            print()
            break

    with open(f"results/{model_name}/{ood_dataset_name}/DEFAULT/test_data/{d}/tokens.dill", "rb") as f:
        x = dill.load(f)
        token_list = tokeniser.batch_decode(x)

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/{loss}/{calibrator}/ood/DEFAULT/{ood_dataset_name}.dill", "rb") as f:
        x = dill.load(f)
        calib_confs = x["calibrated_confs"][n]
        print("CALIB CONF:", calib_confs)
        token_confs = x["calibrated_token_probs"][n]

    print(len(token_list))
    print(len(token_confs))
    for t, c in zip(token_list, token_confs):
        print(f"\\confnew{{{round(c.item(), 2)}}}{{{t}}}{{5}}")

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/{loss}/{calibrator}/calib_weights.dill", "rb") as f:
        x = dill.load(f)
        print(x)
else:
    for i in range(100,400):
        with open(f"results/{model_name}/{dataset_name}/DEFAULT/test_data/data.dill", "rb") as f:
            x = dill.load(f)
            correct = x["correct"][i]
            if correct == 1:
                continue
            n = i
            d = str(n).zfill(4)
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

            break

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/test_data/{d}/tokens.dill", "rb") as f:
        x = dill.load(f)
        token_list = tokeniser.batch_decode(x)

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/test_data/{d}/token_probs.dill", "rb") as f:
        x = dill.load(f)
        original_token_probs = x

    with open(f"results/{model_name}/{dataset_name}/DEFAULT/{loss}/{calibrator}/test_results.dill", "rb") as f:
        x = dill.load(f)
        #print(x.keys())
        print("CALIBRATED CONF:", x["calibrated_confs"][n])
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
