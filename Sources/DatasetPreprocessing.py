import logging
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(logger: logging.Logger,
                    classes_path: str,
                    edges_path: str,
                    features_path: str,
                    save_path: str,
                    seed: int,
                    train_test_split_ratio: float,
                    test_validation_split_ratio: float) -> None:
    logger_prefix = "[preprocess_data]"
    logger.info(f"{logger_prefix}\nstart preprocessing for:\nclasses: {classes_path}\nedges: {edges_path}\n"
                f"features: {features_path}")

    # read csv
    logger.info(f"{logger_prefix} reading csv")
    df_features = pd.read_csv(features_path, header=None)
    df_classes = pd.read_csv(classes_path)

    # mapping all classes to int
    logger.info(f"{logger_prefix} mapping classes to int")
    df_classes['class'] = df_classes['class'].map({'unknown': 0, '1': 1, '2': 0})

    # deleting axis
    logger.info(f"{logger_prefix} deleting axis in classes")
    df_classes = df_classes.drop(columns=["txId"], axis="columns")
    logger.info(f"{logger_prefix} deleting axis in features")
    df_features = df_features.drop(columns=[0], axis="columns")

    # split to train, test, validation
    train_classes, test_classes, train_features, test_features = train_test_split(df_classes,
                                                                                  df_features,
                                                                                  random_state=seed,
                                                                                  train_size=train_test_split_ratio)

    test_classes, validation_classes, test_features, validation_features = \
        train_test_split(test_classes,
                         test_features,
                         random_state=seed,
                         train_size=test_validation_split_ratio)

    logger.info(f"{logger_prefix}\n"
                f"total classes: {len(df_classes)}\n"
                f"total features: {len(df_features)}\n"
                f"train classes: {len(train_classes)}\n"
                f"train features: {len(train_features)}\n"
                f"test classes: {len(test_classes)}\n"
                f"test features: {len(test_features)}\n"
                f"validation classes: {len(validation_classes)}\n"
                f"validation features: {len(validation_features)}")

    # write csv
    logger.info(f"{logger_prefix} writing classes csv")
    train_classes.to_csv(f"{save_path}/train_output.csv", index=False, header=False)
    test_classes.to_csv(f"{save_path}/test_output.csv", index=False, header=False)
    validation_classes.to_csv(f"{save_path}/validation_output.csv", index=False, header=False)

    logger.info(f"{logger_prefix} writing features csv")
    train_features.to_csv(f"{save_path}/train_input.csv", index=False, header=False)
    test_features.to_csv(f"{save_path}/test_input.csv", index=False, header=False)
    validation_features.to_csv(f"{save_path}/validation_input.csv", index=False, header=False)

    logger.info(f"{logger_prefix} end")
