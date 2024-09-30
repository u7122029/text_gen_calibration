import inspect
import sys

from utils import class_predicate
from .apricot import *
from .generic import LogitCalibrator, Calibrator
from .pts import *
from .frequency_pts import *
from .temperature_scaling import TemperatureScaling
from .frequency_ts import *
from .frequency_ts_scaling import *
from .platt_scaling import *
from .token_response_scaler import *
from .lhs_ts import *
from .lhs_fts import *


classes = inspect.getmembers(sys.modules[__name__], class_predicate(Calibrator))
calibrator_dict: dict[str, Calibrator.__class__] = {x: y for x, y in classes}
