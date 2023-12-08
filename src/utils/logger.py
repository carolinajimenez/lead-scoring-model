"""
Custom Logger
"""

import logging

class CustomLogger:
    def __init__(self, name, log_file=None):
        # Configure the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Configure the log format
        formatter = logging.Formatter('\n%(asctime)s - %(levelname)s - %(name)s - %(message)s \n')

        # Configure the console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Configure the file handler if a log file is provided
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_debug(self, message):
        self.logger.debug(message)

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_critical(self, message):
        self.logger.critical(message)

# Example of usage
if __name__ == "__main__":
    # Create an instance of the CustomLogger class
    logger = CustomLogger(name='my_logger', log_file='log_file.txt')

    # Examples of log messages
    logger.log_debug("This is a debug message.")
    logger.log_info("This is an informational message.")
    logger.log_warning("This is a warning message.")
    logger.log_error("This is an error message.")
    logger.log_critical("This is a critical message.")

