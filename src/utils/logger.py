# -*- coding: utf-8 -*-
"""
Custom Logger
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.
import os
import logging


class StringFormatter(logging.Formatter):
    """
    String formatter
    """
    def format(self, record):
        # Convert the log message to a string
        record.msg = str(record.msg)
        return super().format(record)

class CustomLogger:
    def __init__(self, name, log_file=None):
        # Configure the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Configure the log format
        formatter = StringFormatter('%(asctime)s - %(name)s - %(levelname)s: \n%(message)s \n')

        # Configure the console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Configure the file handler if a log file is provided
        if log_file:
            log_folder = 'reports'
            log_file_path = os.path.join(log_folder, log_file)
            file_handler = logging.FileHandler(log_file_path, mode='w')

            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self):
        """
        Get logger
        """
        return self.logger

# Example of usage
if __name__ == "__main__":
    # Create an instance of the CustomLogger class
    logger = CustomLogger(name='my_logger', log_file='test_log.log').get_logger()

    # Examples of log messages
    logger.debug("This is a debug message.")
    logger.info("This is an informational message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

