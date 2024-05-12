import logging

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import numpy as np
from numpy import ndarray

from sklearn import metrics


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
                 plot_width: int,
                 plot_height: int,
                 plot_save_dir: str,
                 prediction_border: float,
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
        self.__plot_width = plot_width
        self.__plot_height = plot_height
        self.__plot_save_dir = plot_save_dir
        self.__prediction_border = prediction_border

        # init None inputs and outputs
        self.__train_input = None
        self.__train_output = None
        self.__test_input = None
        self.__test_output = None
        self.__validation_input = None
        self.__validation_output = None

        # init save file
        self.__save_model_dir = f"{self.__weight_save_dir}/save"
        self.__save_history_file = f"{self.__weight_save_dir}/save/history.save"

        # get compiled model
        self.__model = self.__get_model()

        # flag for avoid unnecessary model loading (plotting after train, for example)
        self.__is_fresh_learned = False

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

    def __save_model(self) -> None:
        self.__model.save(self.__save_model_dir)
        with open(self.__save_history_file, "wb") as f:
            pickle.dump(obj=self.__model.history, file=f)

    def __load_model(self) -> None:
        self.__model = tf.keras.models.load_model(self.__save_model_dir)
        with open(self.__save_history_file, "rb") as f:
            self.__model.history = pickle.load(file=f)
        self.__is_fresh_learned = True

    def __predict_log_proba(self, predictions) -> ndarray:
        return np.array([1 if x >= self.__prediction_border else 0 for x in predictions])

    def __get_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.__input_neurons, activation=self.__activation),
            tf.keras.layers.Dense(units=200,
                                  activation=self.__activation,
                                  activity_regularizer=tf.keras.regularizers.l2(0.015),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.015)),
            tf.keras.layers.Dropout(rate=self.__dropout_rate, seed=self.__seed),
            tf.keras.layers.Dense(units=self.__output_neurons, activation="sigmoid")
        ])
        model.compile(optimizer=self.__optimizer,
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                               tf.keras.metrics.Precision(name="precision", thresholds=0.5),
                               tf.keras.metrics.Recall(name="recall", thresholds=0.5),
                               tf.keras.metrics.F1Score(name="f1", threshold=0.5),
                               tf.keras.metrics.TruePositives(name="TP"),
                               tf.keras.metrics.TrueNegatives(name="TN"),
                               tf.keras.metrics.FalsePositives(name="FP"),
                               tf.keras.metrics.FalseNegatives(name="FN")])
        return model

    def __plot_by_epochs(self, lhs, rhs, lhs_label: str, rhs_label: str, x_label: str, y_label: str, title: str,
                         save_file_name: str):
        fig, ax = plt.subplots()
        ax.plot(lhs, label=lhs_label)
        ax.plot(rhs, label=rhs_label)
        ax.legend()
        fig.set_figwidth(self.__plot_width)
        fig.set_figheight(self.__plot_height)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(f"{self.__plot_save_dir}/{save_file_name}.png")
        plt.show()

    def __plot_history(self) -> None:
        logger_prefix = self.__get_logger_prefix("__plot_history")
        self.__logger.info(f"{logger_prefix} start")

        # loss
        self.__logger.info(f"{logger_prefix} plot loss")
        self.__plot_by_epochs(lhs=self.__model.history.history["loss"],
                              rhs=self.__model.history.history["val_loss"],
                              lhs_label="Training loss",
                              rhs_label="Test loss",
                              x_label="Epochs",
                              y_label="Loss",
                              title="Training and test loss",
                              save_file_name="loss_by_epoch")

        # accuracy
        self.__logger.info(f"{logger_prefix} plot accuracy")
        self.__plot_by_epochs(lhs=self.__model.history.history["accuracy"],
                              rhs=self.__model.history.history["val_accuracy"],
                              lhs_label="Training accuracy",
                              rhs_label="Test accuracy",
                              x_label="Epochs",
                              y_label="Accuracy",
                              title="Training and test accuracy",
                              save_file_name="accuracy_by_epoch")

        # precision
        self.__logger.info(f"{logger_prefix} plot precision")
        self.__plot_by_epochs(lhs=self.__model.history.history["precision"],
                              rhs=self.__model.history.history["val_precision"],
                              lhs_label="Training precision",
                              rhs_label="Test precision",
                              x_label="Epochs",
                              y_label="Precision",
                              title="Training and test precision",
                              save_file_name="precision_by_epoch")

        # recall
        self.__logger.info(f"{logger_prefix} plot recall")
        self.__plot_by_epochs(lhs=self.__model.history.history["recall"],
                              rhs=self.__model.history.history["val_recall"],
                              lhs_label="Training recall",
                              rhs_label="Test recall",
                              x_label="Epochs",
                              y_label="Recall",
                              title="Training and test recall",
                              save_file_name="recall_by_epoch")

        # f1
        self.__logger.info(f"{logger_prefix} plot f1")
        self.__plot_by_epochs(lhs=self.__model.history.history["f1"],
                              rhs=self.__model.history.history["val_f1"],
                              lhs_label="Training f1",
                              rhs_label="Test f1",
                              x_label="Epochs",
                              y_label="F1",
                              title="Training and test f1",
                              save_file_name="f1_by_epoch")

        self.__logger.info(f"{logger_prefix} end")

    def __plot_confusion_matrix(self, confusion_matrix) -> None:
        fig, ax = plt.subplots()
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                    display_labels=["Легальная", "Нелегальная"])
        cm_display.plot(ax=ax, values_format="d")
        fig.set_figwidth(self.__plot_width)
        fig.set_figheight(self.__plot_height)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f"{self.__plot_save_dir}/confusion_matrix.png")
        plt.show()

    def __plot_metrics_bar(self, accuracy: float, precision: float, recall: float, f1: float):
        fig, ax = plt.subplots()

        bars = ax.bar(x=["accuracy", "precision", "recall", "f1"],
                      height=[accuracy, precision, recall, f1])
        ax.bar_label(bars)
        for bars in ax.containers:
            ax.bar_label(bars)

        fig.set_figwidth(self.__plot_width)
        fig.set_figheight(self.__plot_height)
        plt.title("Accuracy, precision, recall, f1")
        plt.savefig(f"{self.__plot_save_dir}/metrics_bar.png")
        plt.show()

        print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\n")

    def __plot_metrics(self) -> None:
        logger_prefix = self.__get_logger_prefix("__plot_metrics")
        self.__logger.info(f"{logger_prefix} start")

        result = self.__model.evaluate(x=self.__validation_input, y=self.__validation_output.to_numpy().flatten())
        accuracy = result[1]
        precision = result[2]
        recall = result[3]
        f1 = result[4][0]
        tp = int(result[5])
        tn = int(result[6])
        fp = int(result[7])
        fn = int(result[8])
        print(result)

        print(f"tp: {tp}; tn: {tn}; fp: {fp}; fn: {fn};")
        confusion_matrix = np.array([[tn, fp], [fn, tp]])

        self.__plot_confusion_matrix(confusion_matrix)
        self.__plot_metrics_bar(accuracy=accuracy, precision=precision, recall=recall, f1=f1)

        self.__logger.info(f"{logger_prefix} end")

    def __test_loop(self) -> None:
        logger_prefix = self.__get_logger_prefix("__test_loop")
        self.__logger.info(f"{logger_prefix} start")

        validation_output_array = self.__validation_output.to_numpy()
        validation_length = len(self.__validation_input)
        predictions = self.__model.predict(self.__validation_input)
        predictions = self.__predict_log_proba(predictions)
        self.__logger.info(f"{logger_prefix}\n"
                           f"Instruction:\n"
                           f"Enter an index from 0 to {validation_length - 1} from validation input to test\n"
                           f"Enter an empty string or not number to stop")

        while True:
            index = input()
            if not index.isnumeric():
                self.__logger.info(f"{logger_prefix} input in not numeric, stopping tests")
                break
            index = int(index)
            if index < 0 or index >= validation_length:
                self.__logger.error(f"{logger_prefix} Invalid index {index}! "
                                    f"Correct values is from 0 to {validation_length - 1}")
            else:
                self.__logger.info(f"Prediction for index {index}: {predictions[index]}. "
                                   f"Correct value is {validation_output_array[index]}")

        self.__logger.info(f"{logger_prefix} end")

    def __learn(self) -> None:
        logger_prefix = self.__get_logger_prefix("__full_learn")
        self.__logger.info(f"{logger_prefix} start")

        self.__model.fit(
            x=self.__train_input,
            y=self.__train_output,
            validation_data=(self.__test_input, self.__test_output),
            epochs=self.__epochs
        )
        self.__is_fresh_learned = True
        self.__save_model()

        self.__logger.info(f"{logger_prefix} end")

    # ==================================================================================================================
    # PUBLIC
    # ==================================================================================================================

    def learn(self) -> None:
        logger_prefix = self.__get_logger_prefix("learn")
        self.__logger.info(f"{logger_prefix} start")

        self.__init_train()
        self.__init_test()
        self.__learn()

        self.__logger.info(f"{logger_prefix} end")

    def plot_results(self) -> None:
        logger_prefix = self.__get_logger_prefix("plot_results")
        self.__logger.info(f"{logger_prefix} start")
        if not self.__is_fresh_learned:
            self.__logger.info(f"{logger_prefix} loading saved model")
            self.__load_model()
        else:
            self.__logger.info(f"{logger_prefix} model is fresh, continue with current weights")
        self.__init_validation()
        self.__plot_history()
        self.__plot_metrics()

        self.__logger.info(f"{logger_prefix} end")

    def start_test(self) -> None:
        logger_prefix = self.__get_logger_prefix("start_test")
        self.__logger.info(f"{logger_prefix} start")

        if not self.__is_fresh_learned:
            self.__logger.info(f"{logger_prefix} loading saved model")
            self.__load_model()
        else:
            self.__logger.info(f"{logger_prefix} model is fresh, continue with current weights")

        self.__init_validation()
        validation_loss, validation_acc = self.__model.evaluate(self.__validation_input, self.__validation_output)
        self.__logger.info(f"{logger_prefix} Validation accuracy: {validation_acc}; Validation loss: {validation_loss}")

        self.__test_loop()

        self.__logger.info(f"{logger_prefix} end")
