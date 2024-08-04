import numpy as np
import scipy.signal as signal


def find_peaks(
        data: np.ndarray,
        sample_rate: int,
        height=None,
        threshold=None,
        prominence=0.5,
        min_frequency=0.08,
) -> np.ndarray:
    """
    Find peaks in the data.
    :param data: Respiratory signal
    :param sample_rate: Sampling rate
    :param height: Required height of peaks
    :param threshold: Required threshold of peaks
    :param prominence: Required prominence of peaks
    :param min_frequency: Minimum respiratory rate in Hz
    :return: Peaks
    """
    distance = min_frequency * sample_rate

    peaks, _ = signal.find_peaks(
        data,
        height=height,
        prominence=prominence,
        threshold=threshold,
        distance=distance)

    return peaks


def frequency_from_peaks(data: np.ndarray, sample_rate: int, height=None, threshold=None, min_frequency=0.08) -> float:
    """
    Peak Counting Method
    :param data:
    :param sample_rate:
    :param height:
    :param threshold:
    :param min_frequency:
    :return:
    """

    peaks = find_peaks(data, sample_rate, height, threshold, min_frequency)
    return len(peaks) / (len(data) / sample_rate)
