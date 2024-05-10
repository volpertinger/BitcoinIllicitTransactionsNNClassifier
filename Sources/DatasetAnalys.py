import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetAnalyser:
    def __init__(self, classes_path: str, edges_path: str, features_path: str,
                 logger: logging.Logger = logging.getLogger()):
        self.__logger_prefix = "[DatasetAnalyser]"
        self.__logger = logger
        self.__classes_path = classes_path
        self.__edges_path = edges_path
        self.__features_path = features_path

        # read csv
        self.__df_classes = pd.read_csv(self.__classes_path)
        self.__df_edges = pd.read_csv(self.__edges_path)
        self.__df_features = pd.read_csv(self.__features_path)

        logger.info(str(self))

    # ==================================================================================================================
    # PRIVATE
    # ==================================================================================================================
    def __get_logger_prefix(self, prefix: str) -> str:
        return f"{self.__logger_prefix} [{prefix}]"

    def __str__(self):
        return f"{self.__logger_prefix}\nclasses_path: {self.__classes_path}\nedges_path: {self.__edges_path}\n" \
               f"features_path: {self.__features_path}"

    # ==================================================================================================================
    # PUBLIC
    # ==================================================================================================================

    def __get_df_heads_str(self):
        return f"{'=' * 120}\n" \
               f"classes\n{self.__df_classes.head()}\n" \
               f"edges\n{self.__df_edges.head()}\n" \
               f"features\n{self.__df_features.head()}\n" \
               f"{'=' * 120}"

    def __plot_by_classes(self):
        group_class = self.__df_classes.groupby('class').count()
        plt.bar(['Illicit', 'Licit', 'Unknown'], group_class['tx_id'].values, color=['r', 'g', 'orange'])
        plt.show()

    def __plot_transactions_by_step(self):
        group_feature = self.__df_features.groupby('time_step').count()
        group_feature['tx_id'].plot()
        plt.title('Number of transactions by time_step')
        plt.show()

    def __plot_transactions_classes_by_time_step(self):
        # Merge Class and features
        df_class_feature = pd.merge(self.__df_classes, self.__df_features)

        group_class_feature = df_class_feature.groupby(['time_step', 'class']).count()
        group_class_feature = group_class_feature['tx_id'].reset_index().rename(columns={'tx_id': 'count'})
        sns.lineplot(x='time_step', y='count', hue='class', data=group_class_feature, palette=['r', 'g', 'orange'])
        plt.show()

        class_illisit = group_class_feature[group_class_feature['class'] == 1]
        class_lisit = group_class_feature[group_class_feature['class'] == 2]
        class_unknown = group_class_feature[group_class_feature['class'] == 3]
        plt.bar(class_unknown['time_step'], class_unknown['count'], color='orange')
        plt.bar(class_lisit['time_step'], class_lisit['count'], color='g',
                bottom=class_unknown['count'])
        plt.bar(class_illisit['time_step'], class_illisit['count'], color='r',
                bottom=np.array(class_unknown['count']) + np.array(class_lisit['count']))
        plt.xlabel('time_step')
        plt.show()

    def analyse(self) -> None:
        logger_prefix = self.__get_logger_prefix("analyse")
        self.__logger.info(f"{logger_prefix} started")

        print(self.__get_df_heads_str())
        self.__plot_by_classes()
        self.__plot_transactions_by_step()
        self.__plot_transactions_classes_by_time_step()

        self.__logger.info(f"{logger_prefix} ended")
