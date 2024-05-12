import torch
from torch import nn
import pandas as pd
import re
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List


class CoTFormat(Enum):
    SYSTEM_USER_CHAT = 1
    USER_CHAT = 2
    NO_TEMPLATE = 3
    DOLLY_15K = 4

    @classmethod
    def from_model_name(cls, name):
        name_dict = {
            "google/gemma-1.1-2b-it": cls.USER_CHAT,
            "google/gemma-1.1-7b-it": cls.USER_CHAT
        }
        return name_dict[name]


class GSMCoT:
    def __init__(self, model_name, token, dset_size=None):
        self.tokeniser = AutoTokenizer.from_pretrained(model_name, token=token, padding_side="left")
        self.tokeniser.pad_token_id = self.tokeniser.eos_token_id

        self.dataset = pd.read_json("data/GSM/test.jsonl", lines=True)

        # Get dataset and ensure that we only get dset_size entries.
        self.dataset["answer"] = self.dataset["answer"].apply(lambda x: int(re.sub(r'[^\w\s]', '', x.split("####")[1])))
        indices = torch.randperm(len(self.dataset))
        if dset_size is not None:
            indices = indices[:dset_size]
        self.dataset = Dataset.from_pandas(self.dataset.iloc[indices.tolist()])

        self.system_text = ("You are a friendly chatbot that only outputs in the form:\n"
                            "**Explanation:** <Your explanation>\n"
                            "**Final Answer:** <A single number>")
        # Format the dataset
        cf = CoTFormat.from_model_name(model_name)
        if cf == CoTFormat.SYSTEM_USER_CHAT:
            format_func = self.__system_user_chat_format
        elif cf == CoTFormat.USER_CHAT:
            format_func = self.__user_chat_format
        elif cf == CoTFormat.NO_TEMPLATE:
            format_func = self.__no_template_format
        else:
            raise Exception(f"Invalid enum value {cf}")

        self.dataset = self.dataset.map(format_func, batched=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map="auto",
                                                          torch_dtype=torch.float16,
                                                          token=token)
        self.__calibrator_model = None
        self.calibrator_type_used = None

    def apply_calibrator(self, calibrator_type, results_batch_size=1, calibration_batch_size=1):
        # TODO: Separate getting generation results from actual calibration
        self.calibrator_type_used = calibrator_type

        dl = DataLoader(self.dataset, batch_size=results_batch_size)
        all_logits = []
        all_eos_masks = []
        all_preds = []
        confs_before_calibration = []
        all_answers = torch.Tensor(self.dataset["answer"])
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
            formatted = batch["formatted"]

            inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
            generated = self.model.generate(**inputs,
                                            max_new_tokens=550,
                                            output_logits=True,
                                            return_dict_in_generate=True,
                                            pad_token_id=self.tokeniser.eos_token_id)
            out_dict = self.__process_generated_output(inputs, generated)
            logits = out_dict["logits"]
            all_logits.append(logits)

            confs_no_calib = out_dict["confidences"]
            final_answers = out_dict["final_answers"]
            eos_masks = out_dict["eos_masks"]
            all_eos_masks.append(eos_masks)

            #all_answers.append(answers)
            all_preds.append(final_answers)
            confs_before_calibration.append(confs_no_calib)

        confs_before_calibration = torch.cat(confs_before_calibration)
        all_preds = torch.cat(all_preds)
        correct = all_preds == all_answers
        calibrator = calibrator_type(self.tokeniser, self.model, False)
        confs_after_calibration, self.__calibrator_model = calibrator.calibrate(
            all_logits=all_logits,
            all_eos_masks=all_eos_masks,
            correct=correct,
            batch_size=calibration_batch_size
        )

        return confs_before_calibration, confs_after_calibration, correct

    def get_calibrated_model(self):
        if self.__calibrator_model is None: return None
        return nn.Sequential(self.model, self.__calibrator_model)

    def __process_generated_output(self, inputs, generated):
        model_logits = torch.stack(generated.logits).permute(1, 0, 2).cpu()
        prob_vecs = torch.softmax(model_logits, dim=2)  # response_idx, response length, vocab_size
        sequences = generated.sequences.cpu()
        responses: torch.Tensor = sequences[:, inputs.input_ids.shape[1]:]
        batch_decoded = self.tokeniser.batch_decode(responses)

        #explanations = []
        final_answers = []
        processed_responses = []
        processed_prob_vecs = []
        eos_masks = []
        processed_logits = []
        response_confidences = []
        for decoded_response, encoded_response, prob_vec, logits in zip(batch_decoded, responses, prob_vecs,
                                                                        model_logits):
            """if remove_eos_tokens:"""
            eos_mask = encoded_response != self.tokeniser.eos_token_id

            logits_no_eos = logits[eos_mask]
            token_response_no_eos = encoded_response[eos_mask]
            prob_vec_no_eos = prob_vec[eos_mask]

            processed_responses.append(token_response_no_eos)
            processed_logits.append(logits_no_eos)

            eos_masks.append(eos_mask)

            token_confidences = torch.take_along_dim(prob_vec_no_eos,
                                                     token_response_no_eos.unsqueeze(1), dim=1).squeeze(1)
            processed_prob_vecs.append(token_confidences)
            response_confidences.append(torch.mean(token_confidences).item())
            """else:
                processed_responses.append(encoded_response)
                token_confidences = torch.take_along_dim(prob_vec, encoded_response.unsqueeze(1), dim=1).squeeze(1)
                processed_prob_vecs.append(token_confidences)
                processed_logits.append(logits)
                response_confidences.append(torch.mean(token_confidences).item())"""

            decoded_response = decoded_response.lower()
            try:
                s1 = decoded_response.split("**explanation:**")[1]
                explanation, final_answer_raw = s1.split("**final answer:**")
                final_answer = int(re.findall(r"\d+", final_answer_raw)[0])
                #explanations.append(explanation)
                final_answers.append(final_answer)
            except:
                #explanations.append("")
                final_answers.append(-1)

        # Computing probabilities using the generated logits.
        out_dict = {
            #"explanations": explanations,
            "tokens": processed_responses,
            "prob_vecs": processed_prob_vecs,
            "logits": model_logits,
            "confidences": torch.Tensor(response_confidences),
            "final_answers": torch.Tensor(final_answers),
            "eos_masks": torch.stack(eos_masks)
        }

        return out_dict

    def __system_user_chat_format(self, x):
        questions = x['question']
        formatted = []
        for question in questions:
            formatted_q = self.tokeniser.apply_chat_template([{"role": "system", "content": self.system_text},
                                                              {"role": "user", "content": f"**Question:** {question}"}],
                                                             tokenize=False,
                                                             add_generation_prompt=True,
                                                             return_tensors="pt")["question"]
            formatted.append(formatted_q)
        return {"formatted": formatted}

    def __user_chat_format(self, x):
        questions = x['question']
        formatted = []
        for question in questions:
            formatted_q = self.tokeniser.apply_chat_template(
                [{"role": "user", "content": f"{self.system_text}\n\n**Question:** {question}"}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            formatted.append(formatted_q)
        return {"formatted": formatted}

    def __no_template_format(self, x):
        questions = x['question']
        formatted = []
        for question in questions:
            formatted_q = f"{self.system_text}\n\n**Question:** {question}\n"
            formatted.append(formatted_q)
        return {"formatted": formatted}