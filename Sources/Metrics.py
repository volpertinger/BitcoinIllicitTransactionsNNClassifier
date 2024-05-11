import math


def accuracy(true_positive: int, true_negative: int, false_positive: int, false_negative: int) -> float:
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


def precision(true_positive: int, false_positive: int) -> float:
    return true_positive / (true_positive + false_positive)


def recall(true_positive: int, false_negative: int) -> float:
    return true_positive / (true_positive + false_negative)


def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    return (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
