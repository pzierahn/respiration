import numpy as np
from typing import Optional
from .feature_point_selection import *
from .feature_point_movement import *


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


def signal_from_amplitudes(point_amplitudes: np.ndarray, use_cgof: bool = False) -> np.ndarray:
    """
    Extract the respiratory signal from the given point amplitudes by averaging the amplitudes of the feature points in
    each frame.
    :param point_amplitudes: The amplitudes of the feature points
    :param use_cgof: Whether to use the correlation-guided optical flow method
    :return:
    """

    # Average the amplitudes of the feature points in each frame
    respiratory_signal = np.sum(point_amplitudes, 1) / point_amplitudes.shape[1]

    if use_cgof:
        respiratory_signal = correlation_guided_optical_flow_method(point_amplitudes, respiratory_signal)

    return respiratory_signal


def extract_signal(
        frames: np.ndarray,
        use_cgof: bool = False,
        roi: Optional[tuple[int, int, int, int]] = None,
        fpn: Optional[int] = None,
        quality_level: float = 0.3,
        quality_level_rv: float = 0.05,
) -> np.ndarray:
    """
    Extract the respiratory signal from the given frames
    :param frames:
    :param use_cgof:
    :param roi:
    :param fpn:
    :param quality_level:
    :param quality_level_rv:
    :return:
    """

    # Extract feature points from the frames
    feature_points = select_feature_points(
        frames[0],
        roi=roi,
        fpn=fpn,
        quality_level=quality_level,
        quality_level_rv=quality_level_rv
    )

    # Track the movement of the feature points
    feature_point_movements = track_feature_point_movement(frames, feature_points)

    # Extract the amplitudes of the feature points
    amplitudes = calculate_feature_point_amplitudes(feature_point_movements)

    return signal_from_amplitudes(amplitudes, use_cgof)
