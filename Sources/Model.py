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

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATv2Conv
from torch.utils.data import random_split
from types import SimpleNamespace
import pickle

warnings.filterwarnings('ignore')


class Model:

    def __init__(self,
                 dataset_save_path: str,
                 weight_save_dir: str,
                 test_split: float,
                 validation_split: float,
                 seed: int,
                 logger: logging.Logger = logging.getLogger()):

        self.__logger_prefix = "[Model]"
        self.__logger = logger
        self.__dataset_save_path = dataset_save_path
        self.__weight_save_dir = weight_save_dir
        self.__test_split = test_split
        self.__validation_split = validation_split
        self.__seed = seed
        self.__dataset: Data = Data()

    # ==================================================================================================================
    # PRIVATE
    # ==================================================================================================================

    def __get_logger_prefix(self, prefix: str) -> str:
        return f"{self.__logger_prefix} [{prefix}]"

    def __load_dataset(self, ) -> Data:
        with open(self.__dataset_save_path, 'rb') as f:
            elliptic_dataset = pickle.load(f)

        print(f'Number of nodes: {elliptic_dataset.num_nodes}')
        print(f'Number of node features: {elliptic_dataset.num_features}')
        print(f'Number of edges: {elliptic_dataset.num_edges}')
        print(f'Number of edge features: {elliptic_dataset.num_features}')
        print(f'Average node degree: {elliptic_dataset.num_edges / elliptic_dataset.num_nodes:.2f}')
        print(f'Number of classes: {len(np.unique(elliptic_dataset.y))}')
        print(f'Has isolated nodes: {elliptic_dataset.has_isolated_nodes()}')
        print(f'Has self loops: {elliptic_dataset.has_self_loops()}')
        print(f'Is directed: {elliptic_dataset.is_directed()}')

        return elliptic_dataset

    def __full_learn(self):
        logger_prefix = self.__get_logger_prefix("__full_learn")
        self.__logger.info(f"{logger_prefix} start")
        if len(self.__dataset) == 0:
            self.__logger.info(f"{logger_prefix} self.__dataset is empty, can`t learn model")
        else:
            # get test, train, validation indexes
            train_idx, test_val_idx = train_test_split(range(self.__dataset.num_nodes),
                                                       random_state=self.__seed,
                                                       shuffle=True,
                                                       test_size=self.__test_split)

            test_idx, val_idx = train_test_split(test_val_idx,
                                                 random_state=self.__seed,
                                                 shuffle=True,
                                                 test_size=self.__validation_split)

            self.__dataset.train_idx = torch.tensor(train_idx, dtype=torch.long)
            self.__dataset.test_idx = torch.tensor(test_idx, dtype=torch.long)
            self.__dataset.val_idx = torch.tensor(val_idx, dtype=torch.long)

            self.__logger.info(f"{logger_prefix}\n"
                               f"total nodes: {self.__dataset.num_nodes}\n"
                               f"train nodes: {len(train_idx)}\n"
                               f"test nodes: {len(test_idx)}\n"
                               f"validation nodes: {len(val_idx)}")

        self.__logger.info(f"{logger_prefix} end")

    # ==================================================================================================================
    # PUBLIC
    # ==================================================================================================================

    def learn(self) -> None:
        logger_prefix = self.__get_logger_prefix("learn")
        self.__logger.info(f"{logger_prefix} start")

        # preprocessed dataset loading
        self.__logger.info(f"{logger_prefix} preprocessed dataset loading")
        try:
            self.__dataset = self.__load_dataset()
        except Exception as e:
            self.__logger.error(f"{logger_prefix} {e}")
            self.__logger.info(f"{logger_prefix} stop due to exception")

        self.__full_learn()

        self.__logger.info(f"{logger_prefix} end")

    def start_test(self) -> None:
        logger_prefix = self.__get_logger_prefix("start_test")
        self.__logger.info(f"{logger_prefix} start")

        self.__logger.info(f"{logger_prefix} end")
