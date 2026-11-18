import os
import sys

# Ensure include/logger can be imported regardless of execution working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from etl_pipeline.include.logger import logging, logger
except ImportError:
    from include.logger import logging, logger


def error_message_detail(error, error_detail: sys):
    """
    Extracts filename, line number, and error message from sys.exc_info()
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    error_message = f"ETL Exception in [{file_name}] line [{line_number}]: {str(error)}"
    return error_message


class CustomETLException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_details)
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
