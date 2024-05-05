import configparser
import Sources.Logger as Logger

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("settings.ini")

    logger = Logger.get_logger(config["Logger"]["level"], config["Logger"]["format"])

    logger.info("Started at")
