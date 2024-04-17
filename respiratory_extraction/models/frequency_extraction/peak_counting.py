import numpy as np
from scipy.signal import find_peaks


def peak_counting(data: np.ndarray, sample_rate: int, height=None, threshold=None, max_rr=45) -> float:
    """
    Peak Counting Method
    :param data:
    :param sample_rate:
    :param height:
    :param threshold:
    :param max_rr:
    :return:
    """

    distance = 60 / max_rr * sample_rate

    peaks, _ = find_peaks(
        data,
        height=height,
        threshold=threshold,
        distance=distance)

    return len(peaks) / (len(data) / sample_rate)
