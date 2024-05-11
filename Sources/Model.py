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
                 input_neurons: int,
                 output_neurons: int,
                 hidden_neurons: int,
                 seed: int,
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
        self.__input_neurons = input_neurons
        self.__output_neurons = output_neurons
        self.__hidden_neurons = hidden_neurons
        self.__seed = seed

        # init None inputs and outputs
        self.__train_input = None
        self.__train_output = None
        self.__test_input = None
        self.__test_output = None
        self.__validation_input = None
        self.__validation_output = None

        # get compiled model
        self.__model = self.__get_model()

    # ==================================================================================================================
    # PRIVATE
    # ==================================================================================================================

    def __get_logger_prefix(self, prefix: str) -> str:
        return f"{self.__logger_prefix} [{prefix}]"

    def __init_train(self) -> None:
        self.__train_input = pd.read_csv(f"{self.__dataset_save_path}/train_input.csv", header=None)
        self.__train_output = pd.read_csv(f"{self.__dataset_save_path}/train_output.csv", header=None)

    def __init_test(self) -> None:
        self.__test_input = pd.read_csv(f"{self.__dataset_save_path}/test_input.csv", header=None)
        self.__test_output = pd.read_csv(f"{self.__dataset_save_path}/test_output.csv", header=None)

    def __init_validation(self) -> None:
        self.__validation_input = pd.read_csv(f"{self.__dataset_save_path}/validation_input.csv", header=None)
        self.__validation_output = pd.read_csv(f"{self.__dataset_save_path}/validation_output.csv", header=None)

    def __get_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.__input_neurons, activation=self.__activation),
            tf.keras.layers.Dense(units=self.__hidden_neurons, activation=self.__activation),
            tf.keras.layers.Dropout(rate=self.__dropout_rate, seed=self.__seed),
            tf.keras.layers.Dense(units=self.__hidden_neurons, activation=self.__activation),
            tf.keras.layers.Dropout(rate=self.__dropout_rate, seed=self.__seed),
            tf.keras.layers.Dense(units=self.__hidden_neurons, activation=self.__activation),
            tf.keras.layers.Dropout(rate=self.__dropout_rate, seed=self.__seed),
            tf.keras.layers.Dense(units=self.__output_neurons),
            tf.keras.layers.Softmax()
        ])
        model.compile(optimizer=self.__optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def __plot_history(self) -> None:
        logger_prefix = self.__get_logger_prefix("__plot_history")
        self.__logger.info(f"{logger_prefix} start")

        self.__logger.info(f"{logger_prefix} end")

    def __test_loop(self) -> None:
        logger_prefix = self.__get_logger_prefix("__test_loop")
        self.__logger.info(f"{logger_prefix} start")

        validation_length = len(self.__validation_input)
        self.__logger.info(f"{logger_prefix}]n"
                           f"Instruction:\n"
                           f"Enter an index from 0 to {validation_length - 1} from validation input to test"
                           f"Enter an empty string or not number to stop")

        predictions = self.__model.predict(self.__validation_input)

        while True:
            index = input()
            if not index.isnumeric():
                break
            index = int(index)
            if index < 0 or index >= validation_length:
                self.__logger.error(f"{logger_prefix} Invalid index {index}! "
                                    f"Correct values is from 0 to {validation_length - 1}")
            else:
                self.__logger.info(f"Prediction for index {index}: {predictions[index]}. "
                                   f"Correct value is {self.__validation_output[index]}")

        self.__logger.info(f"{logger_prefix} end")

    def __full_learn(self) -> None:
        logger_prefix = self.__get_logger_prefix("__full_learn")
        self.__logger.info(f"{logger_prefix} start")

        self.__model.fit(
            x=self.__train_input,
            y=self.__train_output,
            epochs=self.__epochs
        )

        self.__logger.info(f"{logger_prefix} end")

    # ==================================================================================================================
    # PUBLIC
    # ==================================================================================================================

    def learn(self) -> None:
        logger_prefix = self.__get_logger_prefix("learn")
        self.__logger.info(f"{logger_prefix} start")

        self.__init_train()
        self.__full_learn()
        self.__plot_history()

        self.__logger.info(f"{logger_prefix} end")

    def start_test(self) -> None:
        logger_prefix = self.__get_logger_prefix("start_test")
        self.__logger.info(f"{logger_prefix} start")

        self.__init_validation()

        validation_loss, validation_acc = self.__model.evaluate(self.__validation_input, self.__validation_output)
        self.__logger.info(f"{logger_prefix} Validation accuracy: {validation_acc}; Validation loss: {validation_loss}")

        self.__test_loop()

        self.__logger.info(f"{logger_prefix} end")
