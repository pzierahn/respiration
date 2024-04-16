import numpy as np
from scipy import signal


# TODO: Figure out what this does...
def correlation_guided_optical_flow_method(
        point_amplitudes: np.ndarray,
        respiratory_signal: np.ndarray) -> np.ndarray:
    """
    Apply the correlation-guided optical flow method to the point amplitudes.
    :param point_amplitudes:
    :param respiratory_signal:
    :return:
    """
    point_amplitudes_t = np.array(point_amplitudes).T

    augmented_matrix = np.zeros((point_amplitudes_t.shape[0] + 1, point_amplitudes_t.shape[1]))
    augmented_matrix[0, :] = respiratory_signal
    augmented_matrix[1:, :] = point_amplitudes_t

    correlation_matrix = np.corrcoef(augmented_matrix)

    cm_mean = np.mean(abs(correlation_matrix[0, 1:]))

    quality_num = np.array(abs(correlation_matrix[0, 1:]) >= cm_mean).sum()
    quality_feature_point_arg = np.array(abs(correlation_matrix[0, 1:]) >= cm_mean).argsort()[0 - quality_num:]

    cgof_matrix = np.zeros((point_amplitudes.shape[0], quality_num))

    for inx in range(quality_num):
        cgof_matrix[:, inx] = point_amplitudes[:, quality_feature_point_arg[inx]]

    return np.sum(cgof_matrix, 1) / quality_num


def butterworth_filter(
        respiratory_signal: np.ndarray,
        fps: float,
        lowpass: float,
        highpass: float,
        order=int(3)) -> np.ndarray:
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
    Normalize the respiratory signal to a range of -0.5 to 0.5.
    :param respiratory_signal:
    :return:
    """
    max_ampl = max(respiratory_signal)
    min_ampl = min(respiratory_signal)
    return (respiratory_signal - min_ampl) / (max_ampl - min_ampl) - 0.5
