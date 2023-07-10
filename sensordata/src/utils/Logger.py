import datetime
import logging
import os
import sys

from jsonformatter import JsonFormatter

"""
Converts the timestamp in UTC format in the logger file (time).
"""
CUSTOM_FORMAT = {"asctime": lambda: datetime.datetime.now()}

"""
Format of String logger used for logging purpose.
"""
STRING_FORMAT = "%(asctime)s - %(name)s  - %(levelname)s - %(message)s"

"""
Format of JSON logger used for logging purpose.
"""
JSON_FORMAT = """{
    "time":           "%(asctime)s",
    "name":            "name",
    "levelname":       "levelname",
    "message":         "message"
}"""


def logger(fileName: str) -> logging.log:
    """
    Creates and returns Singleton instance of python.

    Arguments:
        name {str} -- name of the file from where the method hs been called.

    Returns:
        system logger : used to log the system logs, like time of methods,
                        some print statement inside a method etc.
    """

    class Logger:
        def __init__(self):
            ### Using streamhandler, it generates logs as standard error.
            self.streamHandler = logging.StreamHandler(stream=sys.stdout)
            self.streamHandler.setFormatter(
                JsonFormatter(JSON_FORMAT, record_custom_attrs=CUSTOM_FORMAT)
            )

        def __call__(self, name):
            ## extracting name from file
            name = os.path.basename(name)

            log = logging.getLogger(name)
            log.setLevel("DEBUG")

            log.addHandler(self.streamHandler)

            return log

    ### Singleton instance Logger would be better here, don't want to generate handlers again.
    logger = Logger()(fileName)

    return logger
