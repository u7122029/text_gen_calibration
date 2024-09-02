import pandas as pd
import torch

from calibrators.frequency_ts import FrequencyTS
from calibrators.universal_calibration_models import TieredPTSModel
from data import DictDataset
from utils import dill_load


class FrequencyPTS(FrequencyTS):
    """
    Calibrates a model by using token confidences across all responses.

    Make sure to initialise this class, then either load() or calibrate() the model.
    """
    def __init__(self, llm_bundle, top_k=10, bot_k=10):
        super().__init__(llm_bundle, top_k, bot_k, TieredPTSModel())

    def load(self, filepath):
        super().load(filepath)
        try:
            self.calibrator_model.ready = True
        except:
            pass