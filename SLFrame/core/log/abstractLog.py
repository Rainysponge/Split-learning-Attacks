"""
log方法的基类，应该包含info, error, debug, warning
"""
import abc
import logging


class AbstractLog(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def info(self, message):
        pass

    @abc.abstractmethod
    def error(self, message):
        pass

    @abc.abstractmethod
    def debug(self, message):
        pass

    @abc.abstractmethod
    def warning(self, message):
        pass

    @abc.abstractmethod
    def critical(self, message):
        pass

    @abc.abstractmethod
    def save(self):
        pass
