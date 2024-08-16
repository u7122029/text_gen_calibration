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

    def get_eval_data_from_dset(self,
                                dset: DictDataset,
                                storage_root: Path,
                                #correctness_func, #
                                #prompt_formatter: PromptFormat, #
                                batch_size=1,
                                max_new_tokens=550,
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
        all_logits_paths = []
        all_tokens_paths = []
        all_logit_confs = []
        #all_preds_successful = []
        #all_preds = []

        dl = DataLoader(dset, batch_size=batch_size)

        # Logits and Output Tokens
        file_idx = 0
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc=desc):
            formatted = batch["response_formatted"]
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
                tokens = response[eos_mask]

                idx_name = str(file_idx).zfill(4)
                logits_path = storage_root / idx_name / f"logits.dill"
                dill_save(processed_logits, logits_path)

                tokens_path = storage_root / idx_name / f"tokens.dill"
                dill_save(tokens, tokens_path)

                all_logits_paths.append(logits_path)
                all_tokens_paths.append(tokens_path)
                file_idx += 1

                prob_vecs = torch.softmax(processed_logits, dim=1)  # response_idx, response length, vocab_size

                token_confidences = torch.take_along_dim(prob_vecs,
                                                         tokens.unsqueeze(1),
                                                         dim=1).squeeze(1)
                response_confidence = torch.mean(token_confidences).item()
                all_logit_confs.append(response_confidence)

                # obtain answer and whether the obtaining was successful.
                """decoded_response = self.tokeniser.decode(tokens)
                decoded_response = decoded_response.lower()

                final_answer, successful = prompt_formatter.obtain_answers(decoded_response)

                all_preds.append(final_answer)
                all_preds_successful.append(successful)"""

        dset = dset.update({"logits": all_logits_paths,
                            "logits_confs": all_logit_confs,
                            "tokens": all_tokens_paths})#,
                            #"pred_successful": all_preds_successful,
                            #"correct": correctness_func(all_preds, dset["answer"])})

        return dset

    def get_verbalised_confs_from_dset(self, dset: DictDataset, batch_size=1, max_new_tokens=30, desc=None):
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
