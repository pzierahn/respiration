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


def find_crossing_points(data: np.ndarray) -> list[int]:
    """
    Find the crossing points
    :return:
    """

    cross_points = []
    for inx in range(len(data) - 1):
        if data[inx] == 0:
            cross_points.append(inx)
        elif data[inx] * data[inx + 1] < 0:
            cross_points.append(inx)

    return cross_points


def find_crossing_points_nfcp(
        data: np.ndarray,
        sample_rate: int,
        quality_level: float = 0.6
) -> list[int]:
    """
    Calculate the frequency from the negative feedback crossover point method
    :param data:
    :param sample_rate:
    :param quality_level:
    :return:
    """

    cross_curve = build_cross_curve(data, sample_rate)
    points = find_crossing_points(cross_curve)
    point_count = len(points)

    if point_count <= 1:
        return points

    # Time span of an average respiratory cycle
    time_span = len(data) / point_count * quality_level

    zero_span = []
    for inx in range(point_count - 1):
        zero_span.append(points[inx + 1] - points[inx])

    while min(zero_span) < time_span:
        doubt_point = np.argmin(zero_span)
        points.pop(doubt_point)

        if len(points) <= 1:
            break

        zero_span = []
        for inx in range(len(points) - 1):
            zero_span.append(points[inx + 1] - points[inx])

    return points


def frequency_from_crossing_point(data: np.ndarray, sample_rate: int) -> float:
    """
    Calculate the frequency from the crossing points
    :return:
    """

    cross_curve = build_cross_curve(data, sample_rate)
    points = find_crossing_points(cross_curve)
    return (len(points) / 2) / (len(data) / sample_rate)


def frequency_from_nfcp(
        data: np.ndarray,
        sample_rate: int,
        quality_level: float = 0.6
) -> float:
    """
    Calculate the frequency from the negative feedback crossover point method
    :param data:
    :param sample_rate:
    :param quality_level:
    :return:
    """

    cross_curve = build_cross_curve(data, sample_rate)
    points = find_crossing_points_nfcp(cross_curve, sample_rate, quality_level)
    return (len(points) / 2) / (len(data) / sample_rate)
