import logging
import sys


def get_logger(log_filename='file'):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(f'{log_filename}.log')

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

