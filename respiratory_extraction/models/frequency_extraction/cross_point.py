import numpy as np


def build_cross_curve(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Build the cross curve
    :return:
    """

    shift_distance = int(sample_rate / 2)
    data_shift = np.zeros(data.shape) - 1
    data_shift[shift_distance:] = data[:-shift_distance]
    return data - data_shift


def crossing_point(data: np.ndarray, sample_rate: int) -> float:
    """
    Crossing Point Method
    :return:
    """

    cross_curve = build_cross_curve(data, sample_rate)

    zero_number = 0
    for inx in range(len(cross_curve) - 1):
        if cross_curve[inx] == 0:
            zero_number += 1
        elif cross_curve[inx] * cross_curve[inx + 1] < 0:
            zero_number += 1

    return (zero_number / 2) / (len(data) / sample_rate)


def negative_feedback_crossover_point_method(
        data: np.ndarray,
        sample_rate: int,
        quality_level=float(0.6)
) -> float:
    cross_curve = build_cross_curve(data, sample_rate)

    zero_number = 0
    zero_index = []
    for inx in range(len(cross_curve) - 1):
        if cross_curve[inx] == 0:
            zero_number += 1
            zero_index.append(inx)
        elif cross_curve[inx] * cross_curve[inx + 1] < 0:
            zero_number += 1
            zero_index.append(inx)

    rr_tmp = ((zero_number / 2) / (len(data) / sample_rate))

    if len(zero_index) <= 1:
        return rr_tmp

    time_span = 60 / rr_tmp / 2 * sample_rate * quality_level
    zero_span = []
    for inx in range(len(zero_index) - 1):
        zero_span.append(zero_index[inx + 1] - zero_index[inx])

    while min(zero_span) < time_span:
        doubt_point = np.argmin(zero_span)
        zero_index.pop(doubt_point)
        zero_index.pop(doubt_point)
        if len(zero_index) <= 1:
            break
        zero_span = []
        for inx in range(len(zero_index) - 1):
            zero_span.append(zero_index[inx + 1] - zero_index[inx])

    return (zero_number / 2) / (len(data) / sample_rate)
