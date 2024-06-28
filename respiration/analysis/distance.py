import numpy as np
import scipy.stats as stats
from scipy.spatial import distance
from dtaidistance import dtw


def distance_euclidean(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two signals.
    :param signal_a: First signal.
    :param signal_b: Second signal.
    :return: The Euclidean distance between the two signals.
    """
    return distance.euclidean(signal_a, signal_b)


def distance_mse(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """
    Calculate the mean squared error between two signals.
    :param signal_a: First signal.
    :param signal_b: Second signal.
    :return: The mean squared error between the two signals.
    """
    return np.mean((signal_a - signal_b) ** 2)


def pearson_correlation(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient between two signals.
    :param signal_a: First signal.
    :param signal_b: Second signal.
    :return: The Pearson correlation coefficient between the two signals.
    """
    correlation, _ = stats.pearsonr(signal_a, signal_b)
    return correlation


def dtw_distance(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """
    Calculate the Dynamic Time Warping distance between two signals.
    :param signal_a: First signal.
    :param signal_b: Second signal.
    :return: The Dynamic Time Warping distance between the two signals.
    """
    return dtw.distance(
        signal_a,
        signal_b,
        window=90,
        use_c=True,
    )
