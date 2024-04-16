import fire
from get_results_pipeline import get_dataset
from chat_formats import prompt_dict
from icecream import ic
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_tokens_and_attentions(word_list, tokeniser):
    items = tokeniser(word_list, add_special_tokens=False, padding=True, return_tensors="pt")
    return items.input_ids, items.attention_mask

def main(prompt_type: str="CoT",
         model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    formatter_cls = prompt_dict[prompt_type]
    with open("token.txt") as f:
        token = f.read().strip()
        ic(token)

    tokeniser = AutoTokenizer.from_pretrained(model_name, token=token, padding_side="left")
    tokeniser.pad_token_id = tokeniser.eos_token_id

    dataset = get_dataset(tokeniser, formatter_cls.format_inputs)
    dl = DataLoader(dataset, batch_size=1)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 token=token)
    stopwords_english = stopwords.words("english")
    stopwords_d = tokeniser(stopwords_english, add_special_tokens=False, padding=True, return_tensors="pt")
    stopword_tokens = stopwords_d.input_ids
    stopword_attention_mask = stopwords_d.attention_mask

    for items in tqdm(dl):
        formatted = items["formatted"]
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
        processed_responses = []
        outputs = torch.zeros(len(responses), 3)

        for i, (response, prob_vec) in enumerate(zip(responses, prob_vecs)):
            mask = torch.ones(len(response))
            eos_mask = torch.ones(len(response))
            eos_indices = torch.where(response == tokeniser.eos_token_id)[0]
            eos_mask[eos_indices] = 0
            eos_mask = eos_mask.bool()

            for stopword_token_set in stopword_tokens:
                response_unfolded = response.unfold(0, len(stopword_token_set), 1)

                indices = torch.where(torch.all(response_unfolded == stopword_token_set, dim=1))[0]
                mask[indices] = 0

            mask = mask.bool()
            non_stopword_tokens = (response[mask])[eos_mask]
            stopword_tokens = (response[not mask])[eos_mask]
            eos_tokens = response[not eos_mask]

            non_stopword_conf = torch.mean(torch.take_along_dim((prob_vec[mask])[eos_mask],
                                                                non_stopword_tokens.unsqueeze(1), dim=1).squeeze(1))
            stopword_conf = torch.mean(torch.take_along_dim((prob_vec[not mask])[eos_mask],
                                                            stopword_tokens.unsqueeze(1), dim=1).squeeze(1))
            eos_conf = torch.mean(torch.take_along_dim(prob_vec[not eos_mask],
                                                       eos_tokens.unsqueeze(1), dim=1).squeeze(1))

            outputs[i, 0] = non_stopword_conf
            outputs[i, 1] = stopword_conf
            outputs[i, 2] = eos_conf
            non_stopword_response, stopword_response, eos_response = (
                tokeniser.batch_decode([non_stopword_tokens, stopword_tokens, eos_tokens]))
            ic(non_stopword_response)
            ic(stopword_response)
            ic(eos_response)
        plt.boxplot(outputs, labels=["x1", "x2", "x3"])
        plt.savefig("output-1.jpg")


if __name__ == "__main__":
    fire.Fire(main)