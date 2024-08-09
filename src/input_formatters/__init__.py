from utils import class_predicate
import inspect
import sys

from .generic import InputFormatter
from .cot_formatters import *


classes = inspect.getmembers(sys.modules[__name__], class_predicate(InputFormatter))
input_formatter_dict: dict[str, InputFormatter.__class__] = {x: y for x, y in classes}
