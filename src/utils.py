from tqdm import tqdm
import torch
from torch.utils.data import Dataset

def generate_over_dataloader(dl, tokeniser, model, max_new_tokens=550, desc=None):
    all_logits = []
    all_tokens = []
    for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=desc):
        formatted = batch["formatted"]

        inputs = tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
        generated = model.generate(**inputs,
                                   max_new_tokens=max_new_tokens,
                                   output_logits=True,
                                   return_dict_in_generate=True,
                                   pad_token_id=tokeniser.eos_token_id)
        model_logits = torch.stack(generated.logits).permute(1, 0, 2).cpu()

        # get the tokens, then remove the ones that made up the input.
        sequences = generated.sequences.cpu()
        responses: torch.Tensor = sequences[:, inputs.input_ids.shape[1]:]

        for logits in model_logits:
            all_logits.append(logits)

        for response in responses:
            all_tokens.append(response)

    out_dict = {
        "all_logits": all_logits,
        "test_tokens": all_tokens
    }
    return out_dict


class TokenLogitsDataset(Dataset):
    def __init__(self, logits, tokens, correct):
        self.logits = logits
        self.tokens = tokens
        self.correct = correct

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
