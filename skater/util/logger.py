"""Funcs for logging"""
import logging


_CRITICAL = logging.CRITICAL
_ERROR = logging.ERROR
_WARNING = logging.WARNING
_INFO = logging.INFO
_DEBUG = logging.DEBUG
_NOTSET = logging.NOTSET


def build_logger(log_level, logger_name, capture_warning=True):
    logger = logging.Logger(logger_name)

    # All warnings are logged by default
    logging.captureWarnings(capture_warning)

    logger.setLevel(log_level)

    msg_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(msg_formatter)
    stream_handler.setFormatter(msg_formatter)
    logger.addHandler(stream_handler)
    return logger
