import unittest
import simple_colors as sc
from input_formatters import GSMCoT, MATHCoT, AQUARATCoT
from prompt_formatters.cot import CoTPromptFormat, PromptVersion
from llm_models.textgen import TextGenLLMBundle


class PromptFormatterDisplayTests(unittest.TestCase):
    """
    Allows the viewing of sample entries from each dataset in CoT format.
    """
    @classmethod
    def setUpClass(cls):
        cls.llm_bundle = TextGenLLMBundle("google/gemma-1.1-2b-it")
        cls.prompt_formatter = CoTPromptFormat(cls.llm_bundle)
        cls.gsmcot = GSMCoT(cls.llm_bundle, PromptVersion.ALT, 1, 1)
        cls.mathcot = MATHCoT(cls.llm_bundle, PromptVersion.ALT, 1, 1)
        cls.aquaratcot = AQUARATCoT(cls.llm_bundle, PromptVersion.ALT, 1, 1)

    """def test_cotprompt_general(self):
        print(sc.red("Testing general CoT prompt format below"))
        print(self.prompt_formatter("What is 2 + 2?"))
        print(sc.blue("Testing CoT numeric conf format") +
              "\n" +
              self.prompt_formatter.conf_format("What is 2 + 2?", "4", "numeric"))
        print(sc.blue("Testing CoT worded conf format") +
              "\n" +
              self.prompt_formatter.conf_format("What is 2 + 2?", "4", "worded"))"""

    def test_cotprompt_gsm(self):
        print(sc.red("Testing GSM CoT prompt format below"))
        print(self.gsmcot.test_dataset["response_formatted"][0])

    def test_cotprompt_math(self):
        print(sc.red("Testing MATH CoT prompt format below"))
        print(self.mathcot.test_dataset["response_formatted"][0])

    def test_cotprompt_aquarat(self):
        print(sc.red("Testing AQUARAT CoT prompt format below"))
        print(self.aquaratcot.test_dataset["response_formatted"][0])


