import configparser
import Sources.Logger as Logger
import Sources.Utils as Utils

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("settings.ini")

    logger = Logger.get_logger(level=config["Logger"]["level"], msg_format=config["Logger"]["format"])
    logger_prefix = "[__main__]"

    logger.info(f"{logger_prefix} Start")

    logger.info(f"{logger_prefix} start data preprocessing")
    Utils.preprocess_data(config=config, logger=logger)

    logger.info(f"{logger_prefix} start data analysing")
    Utils.analyse_data(config=config, logger=logger)

    logger.info(f"{logger_prefix} start model actions")
    Utils.model_actions(config=config, logger=logger)

    logger.info(f"{logger_prefix} end")
