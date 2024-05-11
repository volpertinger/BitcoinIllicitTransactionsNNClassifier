import configparser
import Sources.Logger as Logger
import Sources.DatasetAnalys as DatasetAnalys
import Sources.Model as Model
import Sources.DatasetPreprocessing as DatasetPreprocessing


def preprocess_data(config: configparser.ConfigParser, logger: Logger) -> None:
    if config.getboolean("Actions", "is_need_to_preprocess_dataset"):
        DatasetPreprocessing.preprocess_data(logger=logger,
                                             classes_path=config["Dataset"]["classes_path"],
                                             edges_path=config["Dataset"]["edge_list_path"],
                                             features_path=config["Dataset"]["features_path"],
                                             save_path=config["Saves"]["dataset"],
                                             seed=config.getint("Learn", "seed"),
                                             train_test_split_ratio=config.getfloat("Learn", "train_test_split"),
                                             test_validation_split_ratio=config.getfloat("Learn",
                                                                                         "test_validation_split"))


def analyse_data(config: configparser.ConfigParser, logger: Logger) -> None:
    if config.getboolean("Actions", "is_need_to_analyse_dataset"):
        analyser = DatasetAnalys.DatasetAnalyser(classes_path=config["Dataset"]["classes_path"],
                                                 edges_path=config["Dataset"]["edge_list_path"],
                                                 features_path=config["Dataset"]["features_path"],
                                                 plot_save_dir=config["Saves"]["analyse"],
                                                 plot_width=config.getint("Plot", "width"),
                                                 plot_height=config.getint("Plot", "height"),
                                                 logger=logger)
        analyser.analyse()


def model_actions(config: configparser.ConfigParser, logger: Logger) -> None:
    model = Model.Model(dataset_dir=config["Saves"]["dataset"],
                        weight_save_dir=config["Saves"]["weights"],
                        epochs=config.getint("Learn", "max_epochs"),
                        dropout_rate=config.getfloat("Learn", "dropout_rate"),
                        activation=config["Learn"]["activation"],
                        optimizer=config["Learn"]["optimizer"],
                        input_neurons=config.getint("Learn", "input_neurons"),
                        output_neurons=config.getint("Learn", "output_neurons"),
                        hidden_neurons=config.getint("Learn", "hidden_neurons"),
                        seed=config.getint("Learn", "seed"),
                        plot_save_dir=config["Saves"]["model"],
                        plot_width=config.getint("Plot", "width"),
                        plot_height=config.getint("Plot", "height"),
                        logger=logger)

    if config.getboolean("Actions", "is_need_to_learn"):
        model.learn()

    if config.getboolean("Actions", "is_need_to_plot_learn_results"):
        model.plot_results()

    if config.getboolean("Actions", "is_need_to_test"):
        model.start_test()
