from input_formatters.gsmcot import InputFormatter, GSMCoT
from utils import (TextGenLLMBundle,
                   class_predicate)
import inspect
import sys

classes = inspect.getmembers(sys.modules[__name__], class_predicate(InputFormatter))
input_formatter_dict: dict[str, InputFormatter.__class__] = {x: y for x, y in classes}

if __name__ == "__main__":
    x = GSMCoT(TextGenLLMBundle("google/gemma-1.1-2b-it"), 100, 100)
    print(x.calib_dataset[0].keys())