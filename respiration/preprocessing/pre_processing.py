import numpy as np
from scipy import signal


def butterworth_filter(
        respiratory_signal: np.ndarray,
        fps: float,
        lowpass: float,
        highpass: float,
        order: int = 3) -> np.ndarray:
    """
    Apply a Butterworth filter to the signal.
    :param respiratory_signal: The signal data.
    :param fps: The frames per second.
    :param lowpass: The lowpass frequency in Hz.
    :param highpass: The highpass frequency in Hz.
    :param order: The order of the filter.
    :return:
    """
    b, a, *_ = signal.butter(
        order,
        [2 * lowpass / fps, 2 * highpass / fps],
        output='ba',
        btype='bandpass')
    return signal.filtfilt(b, a, respiratory_signal)


def normalize_signal(time_series: np.ndarray) -> np.ndarray:
    """
    Normalize the respiratory signal
    :param time_series:
    :return:
    """
    return (time_series - np.mean(time_series)) / np.std(time_series)
