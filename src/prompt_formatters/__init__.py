from abc import ABC, abstractmethod
from enum import Enum

from llm_models import TextGenLLMBundle


class CoTModelConfig(Enum):
    SYSTEM_USER_CHAT = 0
    USER_CHAT = 1
    NO_TEMPLATE = 2

    @classmethod
    def from_model_name(cls, name):
        name_dict = {
            "google/gemma-1.1-2b-it": cls.USER_CHAT, # legacy model already downloaded on home machine for quick testing.

            # confirmed models.
            "google/gemma-2-2b-it": cls.USER_CHAT,
            "google/gemma-2-9b-it": cls.USER_CHAT,
            "meta-llama/Meta-Llama-3-8B-Instruct": cls.SYSTEM_USER_CHAT,
            "mistralai/Mistral-7B-Instruct-v0.3": cls.USER_CHAT,
            "microsoft/Phi-3-small-128k-instruct": cls.USER_CHAT,
            "microsoft/Phi-3-mini-128k-instruct": cls.USER_CHAT

            # defunct models.
            # "HuggingFaceH4/zephyr-7b-beta": cls.SYSTEM_USER_CHAT,
            # "01-ai/Yi-1.5-9B-Chat": cls.SYSTEM_USER_CHAT,
            # "NousResearch/Hermes-2-Theta-Llama-3-8B": cls.SYSTEM_USER_CHAT,
            # "NousResearch/Hermes-2-Pro-Mistral-7B": cls.SYSTEM_USER_CHAT,
            # "google/gemma-1.1-7b-it": cls.USER_CHAT,
            # "microsoft/Phi-3-mini-4k-instruct": cls.USER_CHAT
        }
        return name_dict[name]


class PromptFormat(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def conf_format(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, question):
        """
        Formats a question.
        @param question:
        @param kwargs:
        @return:
        """
        pass


class CoTPromptFormat(PromptFormat):
    def __init__(self, llm_bundle: TextGenLLMBundle):
        super().__init__()
        self.llm_bundle = llm_bundle
        self.cot_format = CoTModelConfig.from_model_name(self.llm_bundle.llm_name)

        self.qualitative_scale = {
            "Very low": 0,
            "Low": 0.3,
            "Somewhat low": 0.45,
            "Medium": 0.5,
            "Somewhat high": 0.65,
            "High": 0.7,
            "Very high": 1,
        }
        self.numeric_conf_prompt = ("Provide your confidence in the above answer only as a percentage (0-100%).\n"
                                    "**Confidence:**")
        self.worded_conf_prompt = (f"Provide your confidence in the above answer only as one of "
                                   f"{' / '.join([f'{exp}' for exp in self.qualitative_scale.keys()])}.\n"
                                   f"**Confidence:**")
        self.system_prompt = ("You are a friendly chatbot that only outputs in the form:\n"
                              "**Explanation:** <Your explanation>\n"
                              "**Final Answer:** <A single number>")

        self.final_answer_format = "**Final Answer:** {answer}"
        self.question_format = "**Question:** {question}"

    def conf_format(self, question, answer, conf_prompt_type: str):
        assert conf_prompt_type in {"worded", "numeric"}
        if conf_prompt_type == "worded":
            conf_prompt = self.worded_conf_prompt
        else:
            conf_prompt = self.numeric_conf_prompt

        final_answer_formatted = self.final_answer_format.format(answer)
        question_formatted = self.question_format.format(question)
        if self.cot_format == CoTModelConfig.SYSTEM_USER_CHAT:
            formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                [{"role": "system", "content": f"{question_formatted}\n"
                                               f"{final_answer_formatted}"},
                 {"role": "user", "content": conf_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt")
        elif self.cot_format == CoTModelConfig.USER_CHAT:
            formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                [{"role": "user", "content": f"{question_formatted}\n"
                                             f"{final_answer_formatted}\n\n"
                                             f"{conf_prompt}"}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt")
        else:
            formatted_q = (f"{question_formatted}\n"
                           f"{final_answer_formatted}\n\n"
                           f"{conf_prompt}")

        return formatted_q

    def __call__(self, question):
        question_formatted = self.question_format.format(question=question)
        if self.cot_format == CoTModelConfig.SYSTEM_USER_CHAT:
            formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": question_formatted}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt")
        elif self.cot_format == CoTModelConfig.USER_CHAT:
            formatted_q = self.llm_bundle.tokeniser.apply_chat_template(
                [{"role": "user", "content": f"{self.system_prompt}\n\n"
                                             f"{question_formatted}"}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        else:
            formatted_q = f"{self.system_prompt}\n\n{question_formatted}"

        return formatted_q


class MCQCoTPromptFormat(CoTPromptFormat):
    def __init__(self, llm_bundle: TextGenLLMBundle):
        super().__init__(llm_bundle)
        self.system_prompt = ("You are a friendly chatbot that only outputs in the form:\n"
                              "**Explanation:** <Your explanation>\n"
                              "**Final Answer:** <A single letter>")