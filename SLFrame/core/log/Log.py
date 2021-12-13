"""
log样例
"""
import logging
import time
from .abstractLog import AbstractLog

logger = logging.getLogger()


# logging.basicConfig()


def getTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class Log(AbstractLog):

    def __init__(self, className):
        self.className = className
        self.savePath = ""  # 来自配置文件

    def info(self, message):
        logger.setLevel(logging.INFO)
        logging.info("{}-----{}\n\t{}".format(getTime(), self.className, message))

    def warning(self, message):
        logger.setLevel(logging.WARNING)
        logging.warning("{}-----{}\n\t{}".format(getTime(), self.className, message))

    def error(self, message):
        logger.setLevel(logging.ERROR)
        logging.error("{}-----{}\n\t{}".format(getTime(), self.className, message))

    def debug(self, message):
        logger.setLevel(logging.DEBUG)
        logging.debug("{}-----{}\n\t{}".format(getTime(), self.className, message))

    def critical(self, message):
        logger.setLevel(logging.CRITICAL)
        logging.critical("{}-----{}\n\t{}".format(getTime(), self.className, message))

    def save(self):
        pass
