import logging
from core.log.Log import Log

if __name__ == "__main__":
    log = Log(__name__, {"log_save_path": "log.txt"})
    log.info("hello")
    log.debug("Do something")
    log.warning("Something maybe fail.")
    log.info("Finish")
    # logger = logging.getLogger(__name__)
    # logger.setLevel(level=logging.INFO)
    # handler = logging.FileHandler("log.txt")
    # handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    #
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    #
    # logger.addHandler(handler)
    # logger.addHandler(console)
    #
    # logger.info("Start print log")
    # logger.debug("Do something")
    # logger.warning("Something maybe fail.")
    # logger.info("Finish")
