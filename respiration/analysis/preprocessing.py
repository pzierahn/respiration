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


import numpy as np
from scipy.sparse import spdiags


# TODO: Figure out how this works...
def detrend_tarvainen(signal: np.ndarray, strength: int = 100):
    """
    This function applies a detrending filter.

    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

    :param signal: The signal to be detrended.
    :param strength: The strength of the detrending filter.
    :return: The detrended signal.
    """

    signal_length = signal.shape[0]

    observation_matrix = np.identity(signal_length)

    # second-order difference matrix
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    d = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (observation_matrix - np.linalg.inv(observation_matrix + (strength ** 2) * np.dot(d.T, d))), signal)

    return filtered_signal
