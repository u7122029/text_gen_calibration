import fire
from icecream import ic
from transformers import __version__, AutoTokenizer, AutoModelForCausalLM
from chat_formats import prompt_dict, CoT
from get_results_pipeline import get_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    dl = DataLoader(dataset, batch_size=1)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 #attn_implementation="flash_attention_2",
                                                 resume_download=True,
                                                 token=token)
    correct_pooled = {} # token_idx -> freq
    incorrect_pooled = {} # token_idx -> freq
    for batch in tqdm(dl):


if __name__ == "__main__":
    fire.Fire(main)