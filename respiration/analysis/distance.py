import numpy as np
import scipy.stats as stats

from dtaidistance import dtw
from scipy.spatial import distance
from .peak_counting import find_peaks


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


def pearson_correlation(signal_a: np.ndarray, signal_b: np.ndarray) -> tuple[float, float]:
    """
    Calculate the Pearson correlation coefficient between two signals.
    :param signal_a: First signal.
    :param signal_b: Second signal.
    :return: The Pearson correlation coefficient between the two signals.
    """
    correlation, p = stats.pearsonr(signal_a, signal_b)
    return correlation, p


def spearman_correlation(signal_a: np.array, signal_b: np.array) -> tuple[float, float]:
    """
    Calculate the Spearman correlation
    :param signal_a: First signal.
    :param signal_b: Second signal.
    :return: The Pearson correlation coefficient between the two signals.
    """
    res = stats.spearmanr(signal_a, signal_b)
    return res.statistic, res.pvalue


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


def peak_distance(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        sample_rate: int,
        height=None,
        threshold=None,
        min_frequency=0.08) -> float:
    """
    Calculate the distance between the peaks of the prediction and the ground truth.
    :param prediction:
    :param ground_truth:
    :param sample_rate:
    :param height:
    :param threshold:
    :param min_frequency:
    :return:
    """

    pred_peaks = find_peaks(prediction, sample_rate, height, threshold, min_frequency)
    gt_peaks = find_peaks(ground_truth, sample_rate, height, threshold, min_frequency)

    if len(pred_peaks) == 0 or len(gt_peaks) == 0:
        return np.nan

    return np.mean(np.abs(np.array(pred_peaks) - np.array(gt_peaks)))
