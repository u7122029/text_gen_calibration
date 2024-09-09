from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from data import DictDataset
from llm_models import LLMBundle, extract_verbalized_confidence, VerbalisedConfidence
from utils import HF_TOKEN, dill_save, DEVICE


class TextGenLLMBundle(LLMBundle):
    def get_model(self):
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_name,
                                                              device_map="auto",
                                                              torch_dtype=torch.float16,
                                                              token=HF_TOKEN)
        self.llm_model.eval()
        self.lm_head = self.llm_model.lm_head

        # Freeze all the parameters
        for parameter in self.llm_model.parameters():
            parameter.requires_grad = False

    def get_eval_data_from_dset(self,
                                dset: DictDataset,
                                storage_root: Path,
                                batch_size=1,
                                max_new_tokens=400,
                                desc=None) -> DictDataset:
        """
        Generate the

        - Responses,
        - Logits,
        - Verbalised numerical/quantitative confidences,
        - Verbalised worded/qualitative confidences.
        Over a given dataloader.
        :param dset:
        :param storage_root:
        :param correctness_func:
        :param batch_size:
        :param max_new_tokens:
        :param desc:
        :return:
        """
        print("Getting Evaluation Data.")
        self.load_model()
        all_final_hs_paths = []
        all_tokens_paths = []
        all_logit_confs = []

        dl = DataLoader(dset, batch_size=batch_size)

        # Logits and Output Tokens
        file_idx = 0
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=desc):
            formatted = batch["response_formatted"]
            inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
            generated = self.llm_model.generate(**inputs,
                                                max_new_tokens=max_new_tokens,
                                                #output_logits=True,
                                                output_hidden_states=True,
                                                return_dict_in_generate=True,
                                                pad_token_id=self.tokeniser.eos_token_id)

            # Compute the final hidden state of the model for each token outputted.
            final_hs = torch.stack([h[-1][:, -1, :] for h in generated.hidden_states], dim=1)
            #model_logits = torch.stack(generated.logits).permute(1, 0, 2).cpu()

            #tqdm.write(f"{final_hs.shape}")
            #test_logits = self.lm_head(final_hs).cpu()

            #tqdm.write(f"{torch.norm(test_logits - model_logits)}")

            # get the tokens, then remove the ones that made up the input.
            sequences = generated.sequences.cpu()
            responses: torch.Tensor = sequences[:, inputs.input_ids.shape[1]:].cpu()

            for final_hs_response, response in zip(final_hs, responses): # final_hs used to be model_logits
                eos_mask = response != self.tokeniser.eos_token_id

                processed_final_hs = final_hs_response[eos_mask]
                processed_logits = self.final_hs_to_logits(processed_final_hs.to(DEVICE)).cpu()
                print(processed_logits.shape)
                tokens = response[eos_mask].cpu()

                idx_name = str(file_idx).zfill(4)
                final_hs_path = storage_root / idx_name / f"final_hs.dill"
                dill_save(processed_final_hs, final_hs_path)

                tokens_path = storage_root / idx_name / f"tokens.dill"
                dill_save(tokens, tokens_path)

                all_final_hs_paths.append(final_hs_path)
                all_tokens_paths.append(tokens_path)
                file_idx += 1

                prob_vecs = torch.softmax(processed_logits, dim=1)  # response_idx, response length, vocab_size

                token_confidences = torch.take_along_dim(prob_vecs,
                                                         tokens.unsqueeze(1),
                                                         dim=1).squeeze(1)
                response_confidence = torch.mean(token_confidences).item()
                all_logit_confs.append(response_confidence)

        dset = dset.update({"final_hidden_states": all_final_hs_paths,
                            "logits_confs": all_logit_confs,
                            "tokens": all_tokens_paths})

        return dset

    def get_verbalised_confs_from_dset(self, dset: DictDataset, batch_size=1, max_new_tokens=10, desc=None):
        """

        :param dset:
        :param batch_size:
        :param max_new_tokens:
        :param desc:
        :return:
        """
        out_dict = {
            "numeric_confs": [],
            "numeric_successful": [],
            "worded_confs": [],
            "worded_successful": []
        }

        dl = DataLoader(dset,
                        batch_size=batch_size,
                        collate_fn=dset.collate_fn("numeric_conf_formatted", "worded_conf_formatted"))

        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=desc):
            numeric_formatted = batch["numeric_conf_formatted"]
            worded_formatted = batch["worded_conf_formatted"]

            inputs_numeric = self.tokeniser(numeric_formatted, return_tensors="pt", padding=True).to(DEVICE)
            inputs_worded = self.tokeniser(worded_formatted, return_tensors="pt", padding=True).to(DEVICE)
            numeric_generated = self.llm_model.generate(**inputs_numeric,
                                                        max_new_tokens=max_new_tokens,
                                                        return_dict_in_generate=True,
                                                        pad_token_id=self.tokeniser.eos_token_id)
            worded_generated = self.llm_model.generate(**inputs_worded,
                                                       max_new_tokens=max_new_tokens,
                                                       return_dict_in_generate=True,
                                                       pad_token_id=self.tokeniser.eos_token_id)

            # get the tokens, then remove the ones that made up the input.
            numeric_sequences = numeric_generated.sequences.cpu()
            worded_sequences = worded_generated.sequences.cpu()

            numeric_responses = self.tokeniser.batch_decode(
                numeric_sequences[:, inputs_numeric.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            worded_responses = self.tokeniser.batch_decode(
                worded_sequences[:, inputs_worded.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            n_confidences, n_successful = extract_verbalized_confidence(numeric_responses,
                                                                        VerbalisedConfidence.NUMERIC)
            w_confidences, w_successful = extract_verbalized_confidence(worded_responses,
                                                                        VerbalisedConfidence.WORDED)

            out_dict["numeric_confs"].extend(n_confidences)
            out_dict["numeric_successful"].extend(n_successful)
            out_dict["worded_confs"].extend(w_confidences)
            out_dict["worded_successful"].extend(w_successful)

        out_dict = {k: torch.Tensor(v) for k, v in out_dict.items()}
        out_dict["numeric_successful"] = out_dict["numeric_successful"].bool()
        out_dict["worded_successful"] = out_dict["worded_successful"].bool()

        dset = dset.update(out_dict)
        return dset
