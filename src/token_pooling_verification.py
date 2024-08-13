import fire
from icecream import ic
from transformers import __version__, AutoTokenizer, AutoModelForCausalLM
from chat_formats import prompt_dict, CoT
from metrics import get_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main(prompt_type: str="CoT",
         model_name="google/gemma-1.1-2b-it"):
    ic(__version__)
    formatter_cls = prompt_dict[prompt_type]
    with open("hf_token.txt") as f:
        token = f.read().strip()
        ic(token)

    tokeniser = AutoTokenizer.from_pretrained(model_name, token=token, padding_side="left")
    tokeniser.pad_token_id = tokeniser.eos_token_id
    ic(len(tokeniser))

    dataset = get_dataset(tokeniser,
                          lambda x,y: formatter_cls.format_inputs(x,
                                                                  y,
                                                                  template_type=CoT.ChatTemplateType.USER_CHAT),
                          750)
    dl = DataLoader(dataset, batch_size=2)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 #attn_implementation="flash_attention_2",
                                                 resume_download=True,
                                                 token=token)
    correct_pooled = {} # token_idx -> list of confidences
    incorrect_pooled = {} # token_idx -> list of confidences
    correct_token_responses = {} # token_idx -> list of response numbers
    incorrect_token_responses = {} # token_idx -> list of response numbers
    all_response_confidences = []
    for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
        formatted = batch["formatted"]
        answers = batch["answer"]
        inputs = tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
        generated = model.generate(**inputs,
                                   max_new_tokens=550,
                                   output_logits=True,
                                   return_dict_in_generate=True,
                                   pad_token_id=tokeniser.eos_token_id)
        out_dict = formatter_cls.process_responses(inputs, generated, tokeniser, get_confidence=True)
        """
        out_dict = {
            "explanations": explanations,
            "tokens": responses,
            "final_answers": final_answers,
            "token_confidences": torch.take_along_dim(prob_vecs, responses.unsqueeze(2), dim=2).squeeze(2)
        }"""
        all_response_confidences.append(torch.mean(out_dict["token_confidences"], dim=1))
        correct = answers == out_dict["final_answers"]
        start = batch_idx*dl.batch_size
        end = start + dl.batch_size
        for response_idx, is_correct, token_confs, tokens in zip(range(start, end), correct, out_dict["token_confidences"], out_dict["tokens"]):
            if is_correct.item():
                pooled_dict = correct_pooled
                token_responses_dict = correct_token_responses
            else:
                pooled_dict = incorrect_pooled
                token_responses_dict = incorrect_token_responses

            for token, token_conf in zip(tokens, token_confs):
                token = token.item()
                token_conf = token_conf.item()
                if token not in pooled_dict:
                    pooled_dict[token] = []
                if token not in token_responses_dict:
                    token_responses_dict[token] = set()

                token_responses_dict[token].add(response_idx)
                pooled_dict[token].append(token_conf)
    ic(torch.mean(torch.cat(all_response_confidences)))
    correct_d = {
        "tokens": [],
        "means": [],
        "stds": [],
        "total_occurrences": [],
        "no_responses_appeared": []
    }
    incorrect_d = {
        "tokens": [],
        "means": [],
        "stds": [],
        "total_occurrences": [],
        "no_responses_appeared": []
    }

    correct_pooled_keys = list(correct_pooled.keys())
    correct_d["tokens_str"] = tokeniser.batch_decode(correct_pooled_keys, skip_special_tokens=True)
    for k in correct_pooled_keys:
        v = correct_pooled[k]
        v_tensor = torch.Tensor(v)
        s = correct_token_responses[k]

        correct_d["tokens"].append(k)
        correct_d["total_occurrences"].append(len(v_tensor))
        correct_d["no_responses_appeared"].append(len(s))

        std, mean = torch.std_mean(v_tensor, unbiased=False)
        correct_d["means"].append(mean.item())
        correct_d["stds"].append(std.item())

    incorrect_pooled_keys = list(incorrect_pooled.keys())
    incorrect_d["tokens_str"] = tokeniser.batch_decode(incorrect_pooled_keys, skip_special_tokens=True)
    for k in incorrect_pooled_keys:
        v = incorrect_pooled[k]
        v_tensor = torch.Tensor(v)
        s = incorrect_token_responses[k]

        incorrect_d["tokens"].append(k)
        incorrect_d["total_occurrences"].append(len(v_tensor))
        incorrect_d["no_responses_appeared"].append(len(s))

        std, mean = torch.std_mean(v_tensor, unbiased=False)
        incorrect_d["means"].append(mean.item())
        incorrect_d["stds"].append(std.item())

    correct_df = pd.DataFrame.from_dict(correct_d)
    incorrect_df = pd.DataFrame.from_dict(incorrect_d)

    correct_df["scores"] = ((correct_df["means"] / (1 + correct_df["stds"]))
                            * np.log(correct_df["total_occurrences"])) * np.sqrt(correct_df["no_responses_appeared"])
    incorrect_df["scores"] = ((incorrect_df["means"] / (1 + incorrect_df["stds"]))
                              * np.log(incorrect_df["total_occurrences"])) * np.sqrt(incorrect_df["no_responses_appeared"])

    correct_df = correct_df.sort_values("scores", ascending=True)
    incorrect_df = incorrect_df.sort_values("scores", ascending=True)
    ic(incorrect_df)

    correct_df["sorted_indices"] = range(len(correct_df))
    incorrect_df["sorted_indices"] = range(len(incorrect_df))

    plt.figure()
    plt.bar(correct_df["sorted_indices"], correct_df["scores"], label=correct_df["tokens"])
    plt.title("Scores (Correct Predictions)")

    plt.figure()
    plt.bar(incorrect_df["sorted_indices"], incorrect_df["scores"], label=incorrect_df["tokens"])
    plt.title("Scores (Incorrect Predictions)")

    plt.figure()
    plt.title("Means vs. Scores (Incorrect)")
    plt.scatter(incorrect_df["means"], incorrect_df["scores"])

    plt.figure()
    plt.title("Means vs. Scores (Correct)")
    plt.scatter(correct_df["means"], correct_df["scores"])
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)