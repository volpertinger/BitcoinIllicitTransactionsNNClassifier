import logging


class DatasetAnalyser:
    def __init__(self, logger: logging.Logger = logging.getLogger()):
        self.__logger_prefix = "[DatasetAnalyser]"
        self.__logger = logger

    def __get_logger_prefix(self, prefix: str) -> str:
        return f"{self.__logger_prefix} [{prefix}]"

    def analyse(self) -> None:
        logger_prefix = self.__get_logger_prefix("analyse")
        self.__logger.info(f"{logger_prefix} started")

        self.__logger.info(f"{logger_prefix} ended")
