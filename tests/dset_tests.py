import unittest
from input_formatters import TRIVIAQACoT
from llm_models import TextGenLLMBundle
from prompt_formatters import PromptVersion


class MyTestCase(unittest.TestCase):
    def test_something(self):
        llm_bundle = TextGenLLMBundle("google/gemma-1.1-2b-it")
        x = TRIVIAQACoT(llm_bundle, PromptVersion.DEFAULT, 1, 5)
        print(x.test_dataset[0:2])


if __name__ == '__main__':
    unittest.main()
