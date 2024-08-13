import fire
from metrics import get_dataset
from chat_formats import prompt_dict, CoT
from icecream import ic
from transformers import AutoTokenizer, AutoModelForCausalLM, __version__
import torch
from torch.utils.data import DataLoader
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate


def main(prompt_type: str="CoT",
         model_name: str="mistralai/Mistral-7B-Instruct-v0.2",
         base_max_new_tokens: int=550):
    """
    Do more eos tokens increase or decrease confidence?
    :param prompt_type:
    :param model_name:
    :param base_max_new_tokens:
    :return:
    """
    ic(__version__)
    formatter_cls = prompt_dict[prompt_type]
    with open("hf_token.txt") as f:
        token = f.read().strip()
        ic(token)

    tokeniser = AutoTokenizer.from_pretrained(model_name, token=token, padding_side="left")
    tokeniser.pad_token_id = tokeniser.eos_token_id

    dataset = get_dataset(tokeniser,
                          lambda x,y: formatter_cls.format_inputs(x,
                                                                  y,
                                                                  template_type=CoT.ChatTemplateType.USER_CHAT),
                          100)
    #calib_dataset = get_dataset(tokeniser, None, 720)
    dl = DataLoader(dataset, batch_size=4)

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

    covered = torch.zeros(len(dl.dataset), dtype=torch.bool)
    for base_idx, items in tqdm(zip(range(0, len(dl)*dl.batch_size, dl.batch_size), dl), total=len(dl)):
        formatted = items["formatted"]
        ic(formatted)
        inputs = tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
        generated_base = model.generate(**inputs,
                                        max_new_tokens=base_max_new_tokens,
                                        output_logits=True,
                                        return_dict_in_generate=True,
                                        pad_token_id=tokeniser.eos_token_id)

        # Make another response that contains twice as many eos tokens.
        generated_alt = model.generate(**inputs,
                                       max_new_tokens=base_max_new_tokens*2,
                                       output_logits=True,
                                       return_dict_in_generate=True,
                                       pad_token_id=tokeniser.eos_token_id)

        prob_vecs_base = torch.softmax(torch.stack(generated_base.logits).permute(1, 0, 2), dim=2).cpu()
        sequences_base = generated_base.sequences.cpu()
        responses_base = sequences_base[:, inputs.input_ids.shape[1]:]

        prob_vecs_alt = torch.softmax(torch.stack(generated_alt.logits).permute(1, 0, 2), dim=2).cpu()
        sequences_alt = generated_alt.sequences.cpu()
        responses_alt = sequences_alt[:, inputs.input_ids.shape[1]:]

        # iterate through each generated response within the batch.
        for i, (response_base, prob_vec_base, response_alt, prob_vec_alt) in (
                enumerate(zip(responses_base, prob_vecs_base, responses_alt, prob_vecs_alt))):
            # Find where the <eos> tokens are. Mask should filter them out on application.
            eos_mask_base = torch.ones(len(response_base)) # 0 for <eos>, 1 otherwise
            eos_indices_base = torch.where(response_base == tokeniser.eos_token_id)[0]
            eos_mask_base[eos_indices_base] = 0
            eos_mask_base = eos_mask_base.bool()

            response_alt = torch.ones(len(response_alt))  # 0 for <eos>, 1 otherwise
            eosresponse_alt = torch.where(response_alt == tokeniser.eos_token_id)[0]
            response_alt[eosresponse_alt] = 0
            response_alt = response_alt.bool()

            # Find where the stopwords are. Mask should filter them out on application.
            stopword_mask = torch.ones(len(response_base))  # 0 for stopword, 1 otherwise
            for stopword_token_set, attention_mask in zip(stopword_tokens, stopword_attention_mask):
                tokens = stopword_token_set[attention_mask.bool()]
                response_unfolded = response_base.unfold(0, len(tokens), 1)
                indices = torch.where(torch.all(response_unfolded == tokens, dim=1))[0]
                stopword_mask[indices] = 0

            stopword_mask = stopword_mask.bool()

            non_stopword_tokens = response_base[stopword_mask & eos_mask_base]
            no_eos_tokens = response_base[eos_mask_base]
            extracted_stopword_tokens = response_base[~stopword_mask]
            eos_tokens = response_base[~eos_mask_base]

            regular_conf = torch.mean(torch.take_along_dim(prob_vec_base,
                                                           response_base.unsqueeze(1), dim=1).squeeze(1))
            no_stopword_no_eos_conf = torch.mean(torch.take_along_dim(prob_vec_base[stopword_mask & eos_mask_base],
                                                                non_stopword_tokens.unsqueeze(1), dim=1).squeeze(1))
            no_eos_conf = torch.mean(torch.take_along_dim(prob_vec_base[eos_mask_base],
                                                          no_eos_tokens.unsqueeze(1), dim=1).squeeze(1))
            stopword_conf = torch.mean(torch.take_along_dim(prob_vec_base[~stopword_mask],
                                                            extracted_stopword_tokens.unsqueeze(1), dim=1).squeeze(1))
            eos_conf = torch.mean(torch.take_along_dim(prob_vec_base[~eos_mask_base],
                                                       eos_tokens.unsqueeze(1), dim=1).squeeze(1))

            ic(tokeniser.batch_decode([response_base, non_stopword_tokens, no_eos_tokens]))
            outputs[base_idx + i, 0] = regular_conf
            outputs[base_idx + i, 1] = no_stopword_no_eos_conf
            outputs[base_idx + i, 2] = no_eos_conf
            outputs[base_idx + i, 3] = stopword_conf
            outputs[base_idx + i, 4] = eos_conf
            covered[base_idx + i] = True

    assert torch.all(covered).item()
    eos_confs = outputs[:, 4]
    ic("Means Table:")
    table = [["Unprocessed", "No Stopwords Nor EOS", "No EOS", "Stopwords Only", "EOS Tokens Only"],
             [torch.nanmean(outputs[:, i]) for i in range(5)]]
    print(tabulate(table[1:], headers=table[0], tablefmt="simple_grid"))

    plt.figure()
    plt.boxplot([outputs[:,0], outputs[:, 1], outputs[:, 2], outputs[:, 3], eos_confs[~eos_confs.isnan()]],
                labels=["regular confidences", "no stopwords nor eos",  "no eos", "stopwords only", "eos tokens only"])
    plt.title("Average Token Confidence (ATC) for each CoT Response Type.")
    plt.xticks(rotation=45)
    plt.ylabel("ATC")
    plt.savefig("out.svg", format="svg", bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire(main)