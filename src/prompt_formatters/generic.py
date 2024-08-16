from abc import ABC, abstractmethod


class PromptFormat(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def conf_format(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, question):
        """
        Formats a question.
        @param question:
        @param kwargs:
        @return:
        """
        pass

    @abstractmethod
    def obtain_answers(self, decoded_response):
        pass
