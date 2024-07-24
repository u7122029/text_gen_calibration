from .generic import InputFormatter
from .gsmcot import GSMCoT
from utils import class_predicate
import inspect
import sys

classes = inspect.getmembers(sys.modules[__name__], class_predicate(InputFormatter))
input_formatter_dict: dict[str, InputFormatter.__class__] = {x: y for x, y in classes}
