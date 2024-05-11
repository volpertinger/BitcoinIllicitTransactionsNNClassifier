import logging
import pandas as pd
import tensorflow as tf
import numpy as np


class Model:

    def __init__(self,
                 dataset_dir: str,
                 weight_save_dir: str,
                 epochs: int,
                 dropout_rate: float,
                 activation: str,
                 optimizer: str,
                 logger: logging.Logger = logging.getLogger()):
        # base init from params
        self.__logger_prefix = "[Model]"
        self.__logger = logger
        self.__dataset_save_path = dataset_dir
        self.__weight_save_dir = weight_save_dir
        self.__epochs = epochs
        self.__dropout_rate = dropout_rate
        self.__activation = activation
        self.__optimizer = optimizer

        # get compiled model
        self.__model = self.__get_model()

    # ==================================================================================================================
    # PRIVATE
    # ==================================================================================================================

    def __get_logger_prefix(self, prefix: str) -> str:
        return f"{self.__logger_prefix} [{prefix}]"

    def __get_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(166, activation='relu'),
            tf.keras.layers.Dense(3),
            # tf.keras.layers.Softmax()
        ])
        # TODO настроить compile
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def __full_learn(self) -> None:
        logger_prefix = self.__get_logger_prefix("__full_learn")
        self.__logger.info(f"{logger_prefix} start")

        # train
        train_input = pd.read_csv(f"{self.__dataset_save_path}/train_input.csv", header=None)
        train_output = pd.read_csv(f"{self.__dataset_save_path}/train_output.csv", header=None)
        print(train_input.shape)
        print(train_output.shape)

        self.__model.fit(
            x=train_input,
            y=train_output,
            epochs=self.__epochs
        )

        # validation
        test_input = pd.read_csv(f"{self.__dataset_save_path}/test_input.csv", header=None)
        test_output = pd.read_csv(f"{self.__dataset_save_path}/test_output.csv", header=None)
        print(test_input.shape)
        print(test_output.shape)

        test_loss, test_acc = self.__model.evaluate(test_input, test_output, verbose=2)

        print('\nTest accuracy:', test_acc)

        # predictions
        probability_model = tf.keras.Sequential([self.__model,
                                                 tf.keras.layers.Softmax()])

        predictions = probability_model.predict(test_input)
        print(predictions[0])

        self.__logger.info(f"{logger_prefix} end")

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
