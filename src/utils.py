from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import inspect

RESULTS_PATH = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TextGenLLMBundle:
    def __init__(self, llm_name: str):
        self.llm_name = llm_name

        # Get token.
        with open("token.txt") as f:
            self.token = f.read().strip()

        self.tokeniser = AutoTokenizer.from_pretrained(self.llm_name,
                                                       token=self.token,
                                                       padding_side="left")
        self.tokeniser.pad_token_id = self.tokeniser.eos_token_id
        self.llm_model = None

    def load_model(self):
        """
        Calls the function to load the model into the program. This is a whole separate method because a user might only
        need the tokeniser.
        :return:
        """
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_name,
                                                              device_map="auto",
                                                              torch_dtype=torch.float16,
                                                              token=self.token)

    def is_model_loaded(self):
        return self.llm_model is not None

    def vocab_size(self):
        manual_sizes = {
            "microsoft/Phi-3-mini-4k-instruct": 32064
        }
        if self.llm_name in manual_sizes:
            return manual_sizes[self.llm_name]
        return len(self.tokeniser)

    def generate_over_dataloader(self, dl, max_new_tokens=550, desc=None):

        all_logits = []
        all_tokens = []
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=desc):
            formatted = batch["formatted"]

            inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
            generated = self.llm_model.generate(**inputs,
                                          max_new_tokens=max_new_tokens,
                                          output_logits=True,
                                          return_dict_in_generate=True,
                                          pad_token_id=self.tokeniser.eos_token_id)
            model_logits = torch.stack(generated.logits).permute(1, 0, 2).cpu()

            # get the tokens, then remove the ones that made up the input.
            sequences = generated.sequences.cpu()
            responses: torch.Tensor = sequences[:, inputs.input_ids.shape[1]:]

            for logits, response in zip(model_logits, responses):
                eos_mask = response != self.tokeniser.eos_token_id

                processed_logits = logits[eos_mask]
                processed_response = response[eos_mask]

                all_logits.append(processed_logits)
                all_tokens.append(processed_response)

        return all_logits, all_tokens


class AbsModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class TokenLogitsDataset(Dataset):
    def __init__(self, logits, tokens, correct):
        """

        :param logits: List of tensors. The length of the list is the number of responses, and the shape of each tensor
        is [response_length (num_tokens), vocab_size]
        :param tokens: List of tensors. The length of the list is the number of responses, and the shape of each tensor
        is [response_length (num_tokens)]
        :param correct: Tensor involving boolean values of shape [num_responses].
        """
        self.logits = logits
        self.tokens = tokens
        self.correct = correct

        self.vocab_size = self.logits[0].shape[1]

        assert len(self.logits) == len(self.tokens), \
            f"given logits is not the same length as the tokens. len(logits): {len(self.logits)}, len(tokens): {len(self.tokens)}"
        assert len(self.tokens) == len(self.correct), \
            f"given tokens is not the same length as the labels. len(tokens): {len(self.tokens)}, len(correct): {len(self.correct)}."

        self.correct_vectors = []
        for t, c in zip(self.tokens, self.correct):
            vec = torch.zeros(len(t)) + c
            self.correct_vectors.append(vec)

    def __getitem__(self, item):
        return self.logits[item], self.tokens[item], self.correct_vectors[item]

    def __len__(self):
        return len(self.correct)

    @staticmethod
    def collate_fn(data):
        logits = []
        tokens = []
        correct_vecs = []
        for x in data:
            logits.append(x[0])
            tokens.append(x[1])
            correct_vecs.append(x[2])
        return torch.cat(logits), torch.cat(tokens), torch.cat(correct_vecs)


class TLTokenFrequencyDataset(TokenLogitsDataset):
    def __getitem__(self, item):
        tokens = self.tokens[item]
        relative_tfs = torch.zeros(self.vocab_size)
        token_counts = torch.bincount(tokens)
        relative_tfs[:len(token_counts)] += token_counts
        relative_tfs /= torch.sum(relative_tfs)
        return self.logits[item], tokens, self.correct_vectors[item], relative_tfs

    @staticmethod
    def collate_fn(data):
        logits = []
        tokens = []
        correct_vecs = []
        relative_tfs = []
        for x in data:
            logits.append(x[0])
            tokens.append(x[1])
            correct_vecs.append(x[2])
            relative_tfs.append(x[3])
        return logits, tokens, torch.cat(correct_vecs), torch.stack(relative_tfs)


def get_class_bases(x):
    bases = set()
    for base in x.__bases__:
        bases.add(base)
        bases = bases.union(get_class_bases(base))
    return bases


def class_predicate(cls):
    def predicate_func(x):
        if not inspect.isclass(x): return False

        class_bases = get_class_bases(x)
        return cls in class_bases

    return predicate_func
