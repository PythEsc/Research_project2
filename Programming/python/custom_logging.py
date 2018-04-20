import logging
import sys
from logging import StreamHandler
from typing import Optional


class Logger():
    @staticmethod
    def get(file_path: str, classname, loggername: Optional[str] = None, console_out=sys.stdout):
        log_formatter = logging.Formatter("%(message)s")

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level=logging.DEBUG)
        file_handler.setFormatter(log_formatter)

        console_handler = StreamHandler(console_out)
        console_handler.setLevel(level=logging.INFO)
        console_handler.setFormatter(log_formatter)

        if loggername is None:
            logger = logging.getLogger(classname)
        else:
            logger = logging.getLogger("{} ({})".format(classname, loggername))

        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        pass

    def isatty(self):
        return False
