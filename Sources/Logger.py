import logging


def __configure_logging(level: str, msg_format: str) -> None:
    logging.basicConfig(level=level, format=msg_format)


def get_logger(level, msg_format) -> logging.Logger:
    __configure_logging(level, msg_format)
    logger = logging.getLogger(__name__)
    return logger
