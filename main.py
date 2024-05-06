import configparser
import Sources.Logger as Logger
import Sources.DatasetAnalys as DatasetAnalys
import Sources.Model as Model

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("settings.ini")

    logger = Logger.get_logger(config["Logger"]["level"], config["Logger"]["format"])
    logger_prefix="[__main__]"

    logger.info(f"{logger_prefix} Start")

    if config.getboolean("Actions", "is_need_to_analyse_dataset"):
        analyser = DatasetAnalys.DatasetAnalyser(logger)
        analyser.analyse()

    if config.getboolean("Actions", "is_need_to_learn") or config.getboolean("Actions", "is_need_to_test"):
        model = Model.Model(logger)

        if config.getboolean("Actions", "is_need_to_learn"):
            model.learn()

        if config.getboolean("Actions", "is_need_to_test"):
            model.start_test()

    logger.info(f"{logger_prefix} end")
