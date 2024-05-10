import logging
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import pickle


def preprocess_data(logger: logging.Logger,
                    classes_path: str,
                    edges_path: str,
                    features_path: str,
                    save_path: str):
    logger_prefix = "[preprocess_data]"
    logger.info(f"{logger_prefix} start preprocessing for:\nclasses: {classes_path}\nedges: {edges_path}\n"
                f"features: {features_path}")

    # read csv
    logger.info(f"{logger_prefix} reading csv")
    df_features = pd.read_csv(features_path, header=None)
    df_edges = pd.read_csv(edges_path)
    df_classes = pd.read_csv(classes_path)

    # mapping all classes to int
    logger.info(f"{logger_prefix} mapping classes to int")
    df_classes['class'] = df_classes['class'].map({'unknown': 3, '1': 1, '2': 2})

    # merging node features DF with classes DF
    logger.info(f"{logger_prefix} mapping node features DF with classes DF")
    df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
    df_merge = df_merge.sort_values(0).reset_index(drop=True)

    # mapping nodes to indices
    logger.info(f"{logger_prefix} mapping nodes to indices")
    nodes = df_merge[0].values
    map_id = {j: i for i, j in enumerate(nodes)}

    # mapping edges to indices
    logger.info(f"{logger_prefix} mapping edges to indices")
    edges = df_edges.copy()
    edges.txId1 = edges.txId1.map(map_id)
    edges.txId2 = edges.txId2.map(map_id)
    edges = edges.astype(int)

    edge_index = np.array(edges.values).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

    # weights for the edges are equal in case of model without attention
    weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.float32)

    # mapping node ids to corresponding indexes
    logger.info(f"{logger_prefix} mapping node ids to corresponding indexes")
    node_features = df_merge.drop(['txId'], axis=1).copy()
    node_features[0] = node_features[0].map(map_id)

    labels = node_features['class'].values
    node_features = torch.tensor(np.array(node_features.values, dtype=np.float32), dtype=torch.float32)

    # converting data to PyGeometric graph data format
    logger.info(f"{logger_prefix} converting data to PyGeometric graph data format")
    elliptic_dataset = Data(x=node_features,
                            edge_index=edge_index,
                            edge_attr=weights,
                            y=torch.tensor(labels, dtype=torch.float32))

    logger.info(f"{logger_prefix} saving with pickle")
    with open(save_path, 'wb') as f:
        pickle.dump(elliptic_dataset, f, pickle.HIGHEST_PROTOCOL)

    logger.info(f"{logger_prefix} ended")
