import unittest
from input_formatters import TriviaQACoT
from llm_models import TextGenLLMBundle
from prompt_formatters import PromptVersion


class MyTestCase(unittest.TestCase):
    def test_something(self):
        llm_bundle = TextGenLLMBundle("google/gemma-1.1-2b-it")
        x = TriviaQACoT(llm_bundle, PromptVersion.DEFAULT, 1, 1)
        print(x.test_dataset["response_formatted"][0])


if __name__ == '__main__':
    unittest.main()
