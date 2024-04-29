import inspect
import sys
import torch
from abc import ABC, abstractmethod
from datasets import Dataset
import re
from icecream import ic
from enum import Enum
from tqdm import tqdm


class ChatProcessor(ABC):
    @staticmethod
    @abstractmethod
    def format_inputs(inputs: Dataset, tokeniser, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def process_responses(inputs, model_outs, tokeniser, **kwargs) -> dict:
        pass


class CoT(ChatProcessor):
    class ChatTemplateType(Enum):
        SYSTEM_USER_CHAT = 1
        USER_CHAT = 2
        NO_TEMPLATE = 3
        DOLLY_15K = 4

    @staticmethod
    def format_inputs(inputs: Dataset,
                      tokeniser,
                      template_type: ChatTemplateType = ChatTemplateType.SYSTEM_USER_CHAT):
        out = []
        for question in inputs["question"]:
            system_text = ("You are a friendly chatbot that only outputs in the form:\n"
                           "**Explanation:** <Your explanation>\n"
                           "**Final Answer:** <A single number>")
            if template_type == CoT.ChatTemplateType.SYSTEM_USER_CHAT:
                # Try using the system prompt
                formatted = tokeniser.apply_chat_template([{"role": "system", "content": system_text},
                                                           {"role": "user", "content": f"**Question:** {question}"}],
                                                          tokenize=False,
                                                          add_generation_prompt=True,
                                                          return_tensors="pt")
            elif template_type == CoT.ChatTemplateType.USER_CHAT:
                # Try not using the system prompt
                formatted = tokeniser.apply_chat_template(
                    [{"role": "user", "content": f"{system_text}\n\n**Question:** {question}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                    return_tensors="pt")
            elif template_type == CoT.ChatTemplateType.NO_TEMPLATE:
                formatted = f"{system_text}\n\n**Question:** {question}\n"
            elif template_type == CoT.ChatTemplateType.DOLLY_15K:
                INSTRUCTION_KEY = "### Instruction:"
                RESPONSE_KEY = "### Response:"
                INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                PROMPT_FOR_GENERATION_FORMAT = """{intro}
                {instruction_key}
                {instruction}
                {response_key}
                """.format(
                    intro=INTRO_BLURB,
                    instruction_key=INSTRUCTION_KEY,
                    instruction="{instruction}",
                    response_key=RESPONSE_KEY,
                )
                formatted = PROMPT_FOR_GENERATION_FORMAT.format(instruction=f"{system_text}\n\n**Question:** {question}")

            else:
                raise Exception("Invalid template_type.")
            out.append(formatted)
        return out

    @staticmethod
    def process_responses(inputs, model_outs, tokeniser, **kwargs):
        prob_vecs = torch.softmax(torch.stack(model_outs.logits).permute(1, 0, 2), dim=2).cpu()
        sequences = model_outs.sequences.cpu()
        responses = sequences[:, inputs.input_ids.shape[1]:]
        batch_decoded = tokeniser.batch_decode(responses)

        explanations = []
        final_answers = []
        for response in batch_decoded:
            response = response.lower()
            try:
                s1 = response.split("**explanation:**")[1]
                explanation, final_answer_raw = s1.split("**final answer:**")
                final_answer = int(re.findall(r"\d+", final_answer_raw)[0])
                explanations.append(explanation)
                final_answers.append(final_answer)
            except:
                explanations.append("")
                final_answers.append(-1)

        # Computing probabilities using the generated logits.
        out_dict = {
            "explanations": explanations,
            "tokens": responses,
            "final_answers": torch.Tensor(final_answers),
            "token_confidences": torch.take_along_dim(prob_vecs, responses.unsqueeze(2), dim=2).squeeze(2)
        }

        return out_dict


class CoTAskProb(CoT):
    @staticmethod
    def format_inputs(inputs: Dataset, tokeniser):
        out = []
        for question in inputs["question"]:
            system_text = ("You are a friendly chatbot that only outputs in the form:\n**Explanation:** <Your "
                           "explanation>\n**Final Answer:** <A single number>\n**Probability:** <The probability "
                           "between 0.0 and 1.0 that the final answer is correct>")
            formatted = tokeniser.apply_chat_template([{"role": "system", "content": system_text},
                                                       {"role": "user", "content": question}],
                                                      tokenize=False,
                                                      add_generation_prompt=True,
                                                      return_tensors="pt")
            out.append(formatted)
        return out

    @staticmethod
    def process_responses(inputs, model_outs, tokeniser, **kwargs) -> dict:
        sequences = model_outs.sequences.cpu()
        responses = sequences[:, inputs.input_ids.shape[1]:]
        batch_decoded = tokeniser.batch_decode(responses)

        explanations = []
        final_answers = []
        confidences = []
        for response in batch_decoded:
            try:
                response = response.lower()
                s1 = response.split("**explanation:**")[1]
                s2 = s1.split("**final answer:**")
                explanation = s2[0].strip()
                s3 = s2[1].split("**probability:**")
                final_answer = re.sub(r"[^\d]", "", s3[0])
                confidence = re.findall(r"\d+\.?\d*", s3[1])[0]
                explanations.append(explanation)
                final_answers.append(final_answer)
                confidences.append(confidence)
            except:
                explanations.append("")
                final_answers.append(-1)
                confidences.append(0.5)

        return {
            "explanations": explanations,
            "final_answers": final_answers,
            "confidences": confidences
        }


class FCoT(ChatProcessor):
    @staticmethod
    def format_inputs(inputs: Dataset, tokeniser):
        with open("NL+SL_exemplarset1_prompt.txt") as prompt_file:
            few_shot_text = prompt_file.read()

        with open("NL+SL_exemplarset1_template.txt") as question_file:
            question_text = question_file.read()

        out = []
        for question in inputs["question"]:
            question = question_text.replace("[QUESTION]", question)
            question = f"{few_shot_text}\n\n{question}"
            formatted = tokeniser.apply_chat_template([{"role": "user", "content": question}],
                                                      tokenize=False,
                                                      add_generation_prompt=True,
                                                      return_tensors="pt")
            out.append(formatted)
        return out

    @staticmethod
    def process_responses(inputs,
                          model_outs,
                          tokeniser,
                          end_token="\"\"\"END\"\"\"",
                          get_confidence=False,
                          debug_responses=True) -> dict:
        # inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
        end_tokens = tokeniser(end_token, return_tensors="pt", add_special_tokens=False).input_ids[0]
        prob_vecs = torch.softmax(torch.stack(model_outs.logits).permute(1, 0, 2), dim=2).cpu()
        sequences = model_outs.sequences.cpu()
        responses = sequences[:, inputs.input_ids.shape[1]:]

        truncated_responses = []
        for response in responses:
            response_unfolded = response.unfold(0, len(end_tokens), 1)
            indices = torch.where(torch.all(response_unfolded == end_tokens, dim=1))[0]
            try:
                filter_index = indices[0]
                response_str = response[:filter_index]
                truncated_responses.append(response_str)
            except:
                truncated_responses.append(response)

        batch_decoded = tokeniser.batch_decode(truncated_responses, skip_special_tokens=True)

        explanations = []
        final_answers = []
        for response_str in batch_decoded:
            ldict = {}
            try:
                response_str = response_str.split("\"\"\"END\"\"\"")[0]
            except:
                pass

            explanations.append(response_str)

            if debug_responses:
                ic(response_str)

            try:
                exec(response_str, globals(), ldict)
                answer = ldict["answer"]
                answer = int(answer)
            except:
                answer = -1

            final_answers.append(answer)

        out_dict = {
            "explanations": explanations,
            "tokens": truncated_responses,
            "final_answers": final_answers
        }
        # Computing probabilities using the generated logits.
        if get_confidence:
            compiled_probs = torch.take_along_dim(prob_vecs, responses.unsqueeze(2), dim=2).squeeze(2)
            confidences = compiled_probs.mean(dim=1)
            out_dict["confidences"] = confidences

        return out_dict


def get_class_bases(x):
    bases = set()
    for base in x.__bases__:
        bases.add(base)
        bases = bases.union(get_class_bases(base))
    return bases


def dset_class_predicate(x):
    if not inspect.isclass(x): return False

    class_bases = get_class_bases(x)
    return ChatProcessor in class_bases


classes = inspect.getmembers(sys.modules[__name__], dset_class_predicate)
prompt_dict: dict[str, ChatProcessor] = {x: y for x, y in classes}
