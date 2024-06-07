import numpy as np
from scipy import signal
from scipy.signal import resample


def resample_signal(respiratory_signal: np.ndarray, length: int) -> np.ndarray:
    """
    Resample the signal to the target_fps.
    :param respiratory_signal:
    :param length:
    :return:
    """
    return resample(respiratory_signal, length)


def butterworth_filter(
        respiratory_signal: np.ndarray,
        fps: float,
        lowpass: float,
        highpass: float,
        order: int = 6) -> np.ndarray:
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


def standard_processing(
        respiratory_signal: np.ndarray,
        fps: float = 30.0,
        lowpass: float = 0.08,
        highpass: float = 0.6,
        order: int = 3,
) -> np.ndarray:
    """
    Apply the standard processing to the signal.
    :param respiratory_signal: The signal data.
    :param fps: The frames per second.
    :param lowpass: The lowpass frequency in Hz.
    :param highpass: The highpass frequency in Hz.
    :param order: The order of the filter.
    :return:
    """
    detrend = signal.detrend(respiratory_signal)

    # Apply a Butterworth filter
    filtered_signal = butterworth_filter(
        detrend,
        fps,
        lowpass,
        highpass,
        order
    )

    # Normalize the signal
    return normalize_signal(filtered_signal)
