import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from llm_models import TextGenLLMBundle
from prompt_formatters.generic import PromptFormat


class CoTModelConfig(Enum):
    SYSTEM_USER_CHAT = 0
    USER_CHAT = 1
    NO_TEMPLATE = 2

    @classmethod
    def from_model_name(cls, name):
        name_dict = {
            # confirmed models.
            "google/gemma-2-2b-it": cls.USER_CHAT, # DONE
            "meta-llama/Llama-3.2-3B-Instruct": cls.SYSTEM_USER_CHAT, # DONE
            "Qwen/Qwen2.5-3B-Instruct": cls.SYSTEM_USER_CHAT, # DONE
            "microsoft/Phi-3-mini-4k-instruct": cls.USER_CHAT, # DONE
            "Zyphra/Zamba2-2.7B-instruct": cls.SYSTEM_USER_CHAT, # DONE

            "meta-llama/Llama-3.1-8B-Instruct": cls.SYSTEM_USER_CHAT, # DONE
            "mistralai/Mistral-7B-Instruct-v0.3": cls.USER_CHAT, # DONE
            "Qwen/Qwen2.5-7B-Instruct": cls.SYSTEM_USER_CHAT # DONE
        }
        return name_dict[name]


class CoTPromptFormat(PromptFormat):
    def __init__(self,
                 llm_bundle: TextGenLLMBundle,
                 question_tag="**Question:**",
                 context_tag="**Context:**",
                 final_answer_tag="**Final Answer:**",
                 explanation_tag="**Explanation:**",
                 confidence_tag="**Confidence:**",
                 final_answer_description="The final number obtained from your explanation ONLY, nothing else.",
                 **kwargs):
        super().__init__()
        self.llm_bundle = llm_bundle
        self.cot_format = CoTModelConfig.from_model_name(self.llm_bundle.llm_name)

        self.context_tag = context_tag
        self.question_tag = question_tag
        self.final_answer_tag = final_answer_tag
        self.explanation_tag = explanation_tag
        self.confidence_tag = confidence_tag

        self.final_answer_description = final_answer_description

        self.qualitative_scale = {
            "Very low": 0,
            "Low": 0.3,
            "Somewhat low": 0.45,
            "Medium": 0.5,
            "Somewhat high": 0.65,
            "High": 0.7,
            "Very high": 1,
        } # These confidences are from APRICOT.
        self.numeric_conf_prompt = (f"Provide your confidence in the above answer only as a percentage (0-100%).\n"
                                    f"{self.confidence_tag}")
        self.worded_conf_prompt = (f"Provide your confidence in the above answer only as one of "
                                   f"{' / '.join([f'{exp}' for exp in self.qualitative_scale.keys()])}.\n"
                                   f"{self.confidence_tag}")
        self.system_prompt = (f"You are a chatbot that only outputs in the form:\n"
                              f"{self.explanation_tag} <Your explanation.>\n"
                              f"{self.final_answer_tag} <{self.final_answer_description}>\n")

        self.final_answer_format = f"{self.final_answer_tag} " + "{answer}"
        self.question_format = f"{self.question_tag} " + "{question}"
        self.context_format = f"{self.context_tag} " + "{context}"

    def conf_format(self, question: str, context: Optional[str], answer: str, conf_prompt_type: str):
        """
        Construct a prompt that asks the model how confident its response is given the question and its answer.
        The prompt can be constructed such that it asks for a numeric or worded confidence.
        @param question:
        @param context: Question can have no context.
        @param answer:
        @param conf_prompt_type:
        @return:
        """
        assert conf_prompt_type in {"worded", "numeric"}
        if conf_prompt_type == "worded":
            conf_prompt = self.worded_conf_prompt
        else:
            conf_prompt = self.numeric_conf_prompt

        if context is not None:
            context_formatted = self.context_format.format(context=context) + "\n"
        else:
            context_formatted = ""

        final_answer_formatted = self.final_answer_format.format(answer=answer)
        question_formatted = self.question_format.format(question=question)
        if self.cot_format == CoTModelConfig.SYSTEM_USER_CHAT:
            formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                [{"role": "system", "content": f"{context_formatted}"
                                               f"{question_formatted}\n"
                                               f"{final_answer_formatted}"},
                 {"role": "user", "content": conf_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt")
        elif self.cot_format == CoTModelConfig.USER_CHAT:
            formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                [{"role": "user", "content": f"{context_formatted}"
                                             f"{question_formatted}\n"
                                             f"{final_answer_formatted}\n\n"
                                             f"{conf_prompt}"}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt")
        else:
            formatted_q = (f"{context_formatted}"
                           f"{question_formatted}\n"
                           f"{final_answer_formatted}\n\n"
                           f"{conf_prompt}")

        return formatted_q

    def obtain_answers(self, decoded_responses):
        final_answer_tag_lower = self.final_answer_tag.lower()
        final_preds = []
        all_successful = []
        for decoded_response in decoded_responses:
            decoded_response = decoded_response.lower()
            try:
                _, final_answer_raw = decoded_response.split(final_answer_tag_lower)
                final_prediction = re.findall(r"\d+", final_answer_raw)[0]
                successful = True
            except:
                final_prediction = "-1"  # Indicates a failed response.
                successful = False

            final_preds.append(final_prediction)
            all_successful.append(successful)
        return final_preds, all_successful

    def __call__(self, question, context=None):
        if context is not None:
            context_formatted = self.context_format.format(context=context) + "\n"
        else:
            context_formatted = ""

        question_formatted = self.question_format.format(question=question)
        if self.cot_format == CoTModelConfig.SYSTEM_USER_CHAT:
            formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": f"{context_formatted}\n"
                                             #f"Provide ONLY the exact answer to the following question:\n"
                                             f"{question_formatted}"}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt")
        elif self.cot_format == CoTModelConfig.USER_CHAT:
            formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                [{"role": "user", "content": f"{self.system_prompt}\n\n"
                                             f"{context_formatted}\n"
                                             #f"Provide ONLY the exact answer to the following question:\n"
                                             f"{question_formatted}"}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        else:
            formatted_q = (f"{self.system_prompt}\n\n{context_formatted}\n"
                           #f"Provide ONLY the exact answer to the following question:\n"
                           f"{question_formatted}")

        return formatted_q


class WordAnswerCoTPromptFormat(CoTPromptFormat):
    def __init__(self, llm_bundle, **kwargs):
        super().__init__(llm_bundle,
                         final_answer_description="The exact answer from your explanation ONLY, nothing else.",
                         **kwargs)

    def obtain_answers(self, decoded_responses):
        final_answer_tag_lower = self.final_answer_tag.lower()
        final_preds = []
        all_successful = []
        for decoded_response in decoded_responses:
            decoded_response = decoded_response.lower()
            try:
                _, final_prediction = decoded_response.split(final_answer_tag_lower)
                #final_prediction = re.findall(r"\d+", final_answer_raw)[0]
                successful = True
            except:
                final_prediction = "-1"  # Indicates a failed response.
                successful = False

            final_preds.append(final_prediction)
            all_successful.append(successful)
        return final_preds, all_successful


class MCQCoTPromptFormat(CoTPromptFormat):
    def __init__(self, llm_bundle: TextGenLLMBundle, mcq_options=None, **kwargs):
        if mcq_options is None:
            self.mcq_options = {'a', 'b', 'c', 'd', 'e'}
        else:
            self.mcq_options = set(mcq_options)

        uppers = [x.upper() for x in mcq_options]
        uppers.sort()
        super().__init__(llm_bundle,
                         final_answer_description=f"A single letter indicating your final choice out of [{', '.join(uppers)}] ONLY, nothing else.",
                         **kwargs)


    def obtain_answers(self, decoded_responses):
        final_answer_tag_lower = self.final_answer_tag.lower()
        final_preds = []
        all_successful = []
        for decoded_response in decoded_responses:
            decoded_response = decoded_response.lower()
            try:
                _, final_answer_raw = decoded_response.split(final_answer_tag_lower)
                match = re.search(r'[a-z]', final_answer_raw)
                final_prediction = match.group(0)

                assert final_prediction in self.mcq_options, "go to except."
                successful = True
            except:
                final_prediction = "-1"  # Indicates a failed response.
                successful = False

            final_preds.append(final_prediction)
            all_successful.append(successful)

        return final_preds, all_successful


class AltCoTPromptFormat(CoTPromptFormat):
    """
    Alternative CoT Prompt Format without the *s.
    """
    def __init__(self, llm_bundle: TextGenLLMBundle, **kwargs):
        super().__init__(llm_bundle,
                         "[Question]",
                         "[Context]",
                         "[Conclusion]",
                         "[Reasoning]",
                         "[Certainty]")


class AltWordAnswerCoTPromptFormat(WordAnswerCoTPromptFormat):
    def __init__(self, llm_bundle, **kwargs):
        super().__init__(llm_bundle,
                         question_tag="[Question]",
                         context_tag="[Context]",
                         final_answer_tag="[Conclusion]",
                         explanation_tag="[Reasoning]",
                         confidence_tag="[Certainty]")

    def obtain_answers(self, decoded_responses):
        return WordAnswerCoTPromptFormat.obtain_answers(self, decoded_responses)


class AltMCQCoTPromptFormat(MCQCoTPromptFormat):
    """
    Alternative CoT Prompt Format without the *s.
    """
    def __init__(self, llm_bundle: TextGenLLMBundle, mcq_options=None, **kwargs):
        super().__init__(llm_bundle,
                         mcq_options,
                         question_tag="[Question]",
                         context_tag="[Context]",
                         final_answer_tag="[Conclusion]",
                         explanation_tag="[Reasoning]",
                         confidence_tag="[Certainty]")


class PromptVersion(Enum):
    DEFAULT = 0
    ALT = 1

    @classmethod
    def from_string(cls, name):
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f"{name} is not a valid {cls.__name__}")

    def __call__(self, variant=None):
        mult = 0
        if variant == "mcq":
            mult = 1
        elif variant == "worded":
            mult = 2
        elif variant is not None:
            assert False, f"variant {variant} unrecognised."

        formatters = [CoTPromptFormat,
                      AltCoTPromptFormat,
                      MCQCoTPromptFormat,
                      AltMCQCoTPromptFormat,
                      WordAnswerCoTPromptFormat,
                      AltWordAnswerCoTPromptFormat]
        return formatters[mult * len(self.__class__.__members__) + self.value]