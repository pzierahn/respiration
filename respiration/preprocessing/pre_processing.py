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


def normalize_signal(respiratory_signal: np.ndarray) -> np.ndarray:
    """
    Normalize the respiratory signal
    :param respiratory_signal:
    :return:
    """
    # Normalize the signal between 1 and -1
    return 2 * (respiratory_signal - np.min(respiratory_signal)) / np.ptp(respiratory_signal) - 1


def standard_processing(
        respiratory_signal: np.ndarray,
        fps: float = 30.0,
        lowpass: float = 0.08,
        highpass: float = 0.6,
        order: int = 3,
):
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