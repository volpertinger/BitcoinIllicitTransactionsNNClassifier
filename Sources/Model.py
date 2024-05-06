import logging


class Model:
    def __init__(self, logger: logging.Logger = logging.getLogger()):
        self.__logger_prefix = "[Model]"
        self.__logger = logger

    def __get_logger_prefix(self, prefix: str) -> str:
        return f"{self.__logger_prefix} [{prefix}]"

    def learn(self) -> None:
        logger_prefix = self.__get_logger_prefix("learn")
        self.__logger.info(f"{logger_prefix} started")

        self.__logger.info(f"{logger_prefix} ended")

    def start_test(self) -> None:
        logger_prefix = self.__get_logger_prefix("start_test")
        self.__logger.info(f"{logger_prefix} started")

        self.__logger.info(f"{logger_prefix} ended")
