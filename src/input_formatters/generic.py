from abc import ABC, abstractmethod
from enum import Enum
from typing import Type, Tuple

from calibrators import Calibrator
from data import DictDataset


class InputFormatter(ABC):
    """
    TODO: Determine methods that should be common across all subclasses.
    """

    @abstractmethod
    def __init__(self):
        """
        Abstract constructor to ensure that this class cannot be instantiated.
        """
        pass

    @abstractmethod
    def get_calibration_and_test_data(self, batch_size, recompute=False):
        pass

    @abstractmethod
    def run_calibration_pipeline(self,
                                 calibrator_type: Type[Calibrator],
                                 batch_size=1,
                                 recompute_logits=False,
                                 recalibrate=False,
                                 **kwargs) -> Tuple[DictDataset, DictDataset]:
        pass


class CoTFormat(Enum):
    SYSTEM_USER_CHAT = 0
    USER_CHAT = 1
    NO_TEMPLATE = 2

    @classmethod
    def from_model_name(cls, name):
        name_dict = {
            "google/gemma-1.1-2b-it": cls.USER_CHAT,
            "google/gemma-1.1-7b-it": cls.USER_CHAT,
            "google/gemma-2-9b-it": cls.USER_CHAT,
            "HuggingFaceH4/zephyr-7b-beta": cls.SYSTEM_USER_CHAT,
            "meta-llama/Meta-Llama-3-8B-Instruct": cls.SYSTEM_USER_CHAT,
            "mistralai/Mistral-7B-Instruct-v0.3": cls.USER_CHAT,
            "01-ai/Yi-1.5-9B-Chat": cls.SYSTEM_USER_CHAT,
            "NousResearch/Hermes-2-Theta-Llama-3-8B": cls.SYSTEM_USER_CHAT,
            "NousResearch/Hermes-2-Pro-Mistral-7B": cls.SYSTEM_USER_CHAT,
            "microsoft/Phi-3-small-128k-instruct": cls.USER_CHAT,
            "microsoft/Phi-3-mini-128k-instruct": cls.USER_CHAT,
            "microsoft/Phi-3-mini-4k-instruct": cls.USER_CHAT
        }
        return name_dict[name]
