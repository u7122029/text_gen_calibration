from abc import ABC, abstractmethod


class PromptFormat(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def conf_format(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, question, context):
        """
        Formats a question with context if available.
        @param question:
        @param context:
        @param kwargs:
        @return:
        """
        pass

    @abstractmethod
    def obtain_answers(self, decoded_response):
        pass
