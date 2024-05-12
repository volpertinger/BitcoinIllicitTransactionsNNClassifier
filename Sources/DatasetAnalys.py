import logging
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame


class DatasetAnalyser:
    def __init__(self,
                 classes_path: str,
                 edges_path: str,
                 features_path: str,
                 plot_save_dir: str,
                 plot_width: int,
                 plot_height: int,
                 logger: logging.Logger = logging.getLogger()):
        # init from params
        self.__logger_prefix = "[DatasetAnalyser]"
        self.__logger = logger
        self.__classes_path = classes_path
        self.__edges_path = edges_path
        self.__features_path = features_path
        self.__plot_save_dir = plot_save_dir
        self.__plot_width = plot_width
        self.__plot_height = plot_height

        # read csv
        self.__df_classes: DataFrame = pd.read_csv(self.__classes_path)
        self.__df_edges: DataFrame = pd.read_csv(self.__edges_path)
        self.__df_features: DataFrame = pd.read_csv(self.__features_path)

        # color legend, labels
        self.__colors = {"illicit": "red", "licit": "green"}
        self.__classes_en_to_rus = {"illicit": "Нелегальная", "licit": "Легальная"}

        # prettify csv
        self.__df_prettify()

        logger.info(str(self))

    # ==================================================================================================================
    # PRIVATE
    # ==================================================================================================================
    def __get_logger_prefix(self, prefix: str) -> str:
        return f"{self.__logger_prefix} [{prefix}]"

    def __str__(self) -> str:
        return f"{self.__logger_prefix}\nclasses_path: {self.__classes_path}\nedges_path: {self.__edges_path}\n" \
               f"features_path: {self.__features_path}"

    def __df_prettify(self) -> None:
        # edges
        self.__df_edges = self.__df_edges.rename(columns={"txId1": "tx_id_lhs", "txId2": "tx_id_rhs"})

        # classes
        self.__df_classes = self.__df_classes.rename(columns={"txId": "tx_id"})

        # features
        self.__df_features.columns = ["tx_id"] + \
                                     ["time_step"] + \
                                     [f"local_feature_{i + 1}" for i in range(93)] + \
                                     [f"aggregated_feature_{i + 1}" for i in range(72)]
        self.__df_features = pd.merge(self.__df_classes, self.__df_features)

        # deleting unknown classes
        self.__df_classes = self.__df_classes.drop(self.__df_classes[self.__df_classes["class"] == "unknown"].index)
        self.__df_features = self.__df_features.drop(self.__df_features[self.__df_features["class"] == "unknown"].index)

        # classes
        self.__df_classes["class"] = self.__df_classes["class"].map(
            {"1": self.__classes_en_to_rus["illicit"],
             "2": self.__classes_en_to_rus["licit"],
             "unknown": self.__classes_en_to_rus["licit"]})

        # features
        self.__df_features = self.__df_features.drop(columns=["class"], axis="columns")

    def __get_df_heads_str(self) -> str:
        return f"{'=' * 120}\n" \
               f"classes\n{self.__df_classes.head()}\n" \
               f"edges\n{self.__df_edges.head()}\n" \
               f"features\n{self.__df_features.head()}\n" \
               f"{'=' * 120}"

    def __plot_classes_bar(self) -> None:
        indexes = list(self.__classes_en_to_rus.values())
        values = []
        for index in indexes:
            values.append(self.__df_classes["class"].value_counts()[index])

        # plotting figure
        fig, ax = plt.subplots()
        bars = ax.bar(x=indexes,
                      height=values,
                      color=[self.__colors["illicit"],
                             self.__colors["licit"]])
        ax.bar_label(bars)
        for bars in ax.containers:
            ax.bar_label(bars)

        # prettify, save, show
        fig.set_figwidth(self.__plot_width)
        fig.set_figheight(self.__plot_height)
        plt.xlabel("Класс транзакции")
        plt.ylabel("Количество транзакций")
        plt.title("Transactions by classes bar")
        plt.savefig(f"{self.__plot_save_dir}/classes_bar.png")
        plt.show()

    def __plot_classes_bar_by_time_step(self) -> None:
        # Merge classes and features
        df_class_feature = pd.merge(self.__df_classes,
                                    self.__df_features)

        # grouping
        group_class_feature = df_class_feature.groupby(['time_step', 'class']).count()
        group_class_feature = group_class_feature['tx_id'].reset_index().rename(columns={'tx_id': 'count'})

        # count classes
        class_illisit = group_class_feature[group_class_feature["class"] == self.__classes_en_to_rus["illicit"]]
        class_lisit = group_class_feature[group_class_feature["class"] == self.__classes_en_to_rus["licit"]]

        # plotting
        fig, ax = plt.subplots()
        ax.bar(x=class_lisit['time_step'],
               height=class_lisit['count'],
               color='green')
        ax.bar(x=class_illisit['time_step'],
               height=class_illisit['count'],
               color='red',
               bottom=class_lisit['count'])

        labels = list(self.__colors.keys())
        labels_translated = [self.__classes_en_to_rus[i] for i in labels]
        handles = [plt.Rectangle((0, 0), 1, 1, color=self.__colors[label]) for label in labels]
        plt.legend(handles, labels_translated)

        # prettify, save, show
        fig.set_figwidth(self.__plot_width)
        fig.set_figheight(self.__plot_height)
        plt.xlabel("Временной шаг")
        plt.ylabel("Количество транзакций")
        plt.title("Classes bar by time step")
        plt.savefig(f"{self.__plot_save_dir}/classes_by_timestep_bar.png")
        plt.show()

    # ==================================================================================================================
    # PUBLIC
    # ==================================================================================================================

    def analyse(self) -> None:
        logger_prefix = self.__get_logger_prefix("analyse")
        self.__logger.info(f"{logger_prefix} started")

        print(self.__get_df_heads_str())
        self.__plot_classes_bar()
        self.__plot_classes_bar_by_time_step()

        self.__logger.info(f"{logger_prefix} ended")
