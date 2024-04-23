import fire
from get_results_pipeline import get_dataset
from chat_formats import prompt_dict, CoT
from icecream import ic
from transformers import AutoTokenizer, AutoModelForCausalLM, __version__
import torch
from torch.utils.data import DataLoader
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt


def main(prompt_type: str="CoT",
         model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    ic(__version__)
    formatter_cls = prompt_dict[prompt_type]
    with open("token.txt") as f:
        token = f.read().strip()
        ic(token)

    tokeniser = AutoTokenizer.from_pretrained(model_name, token=token, padding_side="left")
    tokeniser.pad_token_id = tokeniser.eos_token_id

    dataset = get_dataset(tokeniser,
                          lambda x,y: formatter_cls.format_inputs(x,
                                                                  y,
                                                                  template_type=CoT.ChatTemplateType.SYSTEM_USER_CHAT),
                          720)
    #dataset = get_dataset(tokeniser, None, 720)
    dl = DataLoader(dataset, batch_size=1)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 #attn_implementation="flash_attention_2",
                                                 resume_download=True,
                                                 token=token)
    stopwords_original = stopwords.words("english")
    stopwords_english = stopwords_original + [f" {x.capitalize()}" for x in stopwords_original]
    stopwords_english += [f" {x}" for x in stopwords_original]
    stopwords_english += [x.capitalize() for x in stopwords_original]

    stopwords_d = tokeniser(stopwords_english, add_special_tokens=False, padding=True, return_tensors="pt")
    stopword_tokens = stopwords_d.input_ids
    stopword_attention_mask = stopwords_d.attention_mask
    outputs = torch.zeros(len(dl.dataset), 5)

    for base_idx, items in tqdm(zip(range(0, len(dl)*dl.batch_size, dl.batch_size), dl), total=len(dl)):
        formatted = items["formatted"]
        ic(formatted)
        inputs = tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
        generated = model.generate(**inputs,
                                   max_new_tokens=550,
                                   output_logits=True,
                                   return_dict_in_generate=True,
                                   pad_token_id=tokeniser.eos_token_id
                                   )
        prob_vecs = torch.softmax(torch.stack(generated.logits).permute(1, 0, 2), dim=2).cpu()
        sequences = generated.sequences.cpu()
        responses = sequences[:, inputs.input_ids.shape[1]:]

        # iterate through each generated response within the batch.
        for i, (response, prob_vec) in enumerate(zip(responses, prob_vecs)):
            # Find where the <eos> tokens are. Mask should filter them out on application.
            eos_mask = torch.ones(len(response)) # 0 for <eos>, 1 otherwise
            eos_indices = torch.where(response == tokeniser.eos_token_id)[0]
            eos_mask[eos_indices] = 0
            eos_mask = eos_mask.bool()

            # Find where the stopwords are. Mask should filter them out on application.
            stopword_mask = torch.ones(len(response))  # 0 for stopword, 1 otherwise
            for stopword_token_set, attention_mask in zip(stopword_tokens, stopword_attention_mask):
                tokens = stopword_token_set[attention_mask.bool()]
                response_unfolded = response.unfold(0, len(tokens), 1)
                indices = torch.where(torch.all(response_unfolded == tokens, dim=1))[0]
                stopword_mask[indices] = 0

            stopword_mask = stopword_mask.bool()
            not_eos_mask = torch.logical_not(eos_mask)

            mask_and_eos_mask = torch.logical_and(stopword_mask, eos_mask)
            inv_mask_and_eos_mask = torch.logical_not(mask_and_eos_mask)

            non_stopword_tokens = response[mask_and_eos_mask]
            no_eos_tokens = response[eos_mask]
            extracted_stopword_tokens = response[inv_mask_and_eos_mask]
            eos_tokens = response[not_eos_mask]

            regular_conf = torch.mean(torch.take_along_dim(prob_vec,
                                                           response.unsqueeze(1), dim=1).squeeze(1))
            no_eos_conf = torch.mean(torch.take_along_dim(prob_vec[eos_mask],
                                                          no_eos_tokens.unsqueeze(1), dim=1).squeeze(1))
            non_stopword_conf = torch.mean(torch.take_along_dim(prob_vec[mask_and_eos_mask],
                                                                non_stopword_tokens.unsqueeze(1), dim=1).squeeze(1))
            stopword_conf = torch.mean(torch.take_along_dim(prob_vec[inv_mask_and_eos_mask],
                                                            extracted_stopword_tokens.unsqueeze(1), dim=1).squeeze(1))
            eos_conf = torch.mean(torch.take_along_dim(prob_vec[not_eos_mask],
                                                       eos_tokens.unsqueeze(1), dim=1).squeeze(1))

            ic(tokeniser.batch_decode([response, non_stopword_tokens, no_eos_tokens]))
            outputs[base_idx + i, 0] = regular_conf
            outputs[base_idx + i, 1] = non_stopword_conf
            outputs[base_idx + i, 2] = no_eos_conf
            outputs[base_idx + i, 3] = stopword_conf
            outputs[base_idx + i, 4] = eos_conf

    eos_confs = outputs[:, 4]
    plt.figure()
    plt.boxplot([outputs[:,0], outputs[:, 1], outputs[:, 2], outputs[:, 3], eos_confs[~eos_confs.isnan()]],
                labels=["regular confidences", "no eos",  "no stopwords nor eos", "stopwords and eos only", "eos tokens only"])
    plt.title("Average Token Confidence (ATC) for each CoT Response Type.")
    plt.xticks(rotation=45)
    plt.ylabel("ATC")
    plt.savefig("out.svg", format="svg", bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire(main)
