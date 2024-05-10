import logging
import pandas as pd
import networkx as nx
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import torch
import warnings
import seaborn as sns


class Model:

    def __init__(self,
                 dataset_dir: str,
                 weight_save_dir: str,
                 epochs: int,
                 logger: logging.Logger = logging.getLogger()):
        self.__logger_prefix = "[Model]"
        self.__logger = logger
        self.__dataset_save_path = dataset_dir
        self.__weight_save_dir = weight_save_dir
        self.__epochs = epochs
        self.__model = self.__get_model()

    # ==================================================================================================================
    # PRIVATE
    # ==================================================================================================================

    def __get_logger_prefix(self, prefix: str) -> str:
        return f"{self.__logger_prefix} [{prefix}]"

    def __full_learn(self):
        logger_prefix = self.__get_logger_prefix("__full_learn")
        self.__logger.info(f"{logger_prefix} start")

        train_input = pd.read_csv(f"{self.__dataset_save_path}/train_input.csv", header=None)
        train_output = pd.read_csv(f"{self.__dataset_save_path}/train_output.csv", header=None)
        # train_tf = tf.convert_to_tensor(train_input).shape[0]
        # train_tf_o = tf.convert_to_tensor(train_output).shape[0]
        # print(train_tf.shape)
        # print(train_tf_o.shape)
        print(train_input.shape)
        print(train_output.shape)

        self.__model.fit(
            x=train_input,
            y=train_output,
            epochs=self.__epochs
        )

        self.__logger.info(f"{logger_prefix} end")

    def __get_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            # tf.keras.layers.Flatten(input_shape=(163014, 1)),
            tf.keras.layers.Dense(166, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        # TODO настроить compile
        model.compile(loss=tf.keras.losses.categorical_hinge)
        return model

    # ==================================================================================================================
    # PUBLIC
    # ==================================================================================================================

    def learn(self) -> None:
        logger_prefix = self.__get_logger_prefix("learn")
        self.__logger.info(f"{logger_prefix} start")

        self.__full_learn()

        self.__logger.info(f"{logger_prefix} end")

    def start_test(self) -> None:
        logger_prefix = self.__get_logger_prefix("start_test")
        self.__logger.info(f"{logger_prefix} start")

        self.__logger.info(f"{logger_prefix} end")
