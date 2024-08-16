import unittest
import simple_colors as sc
from prompt_formatters import CoTPromptFormat
from llm_models import TextGenLLMBundle


class PromptFormatterDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_bundle = TextGenLLMBundle("google/gemma-1.1-2b-it")
        cls.prompt_formatter = CoTPromptFormat(cls.llm_bundle)

    def test_cotprompt(self):
        print(sc.blue("Testing CoT question format") + "\n" + self.prompt_formatter("What is 2 + 2?"))
        print(sc.blue("Testing CoT numeric conf format") +
              "\n" +
              self.prompt_formatter.conf_format("What is 2 + 2?", "4", "numeric"))
        print(sc.blue("Testing CoT worded conf format") +
              "\n" +
              self.prompt_formatter.conf_format("What is 2 + 2?", "4", "worded"))





