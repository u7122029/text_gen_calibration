import inspect, sys
from utils import class_predicate

from .cot import *
from .generic import *

classes = inspect.getmembers(sys.modules[__name__], class_predicate(PromptFormat))
prompt_formatter_dict: dict[str, PromptFormat.__class__] = {x: y for x, y in classes}