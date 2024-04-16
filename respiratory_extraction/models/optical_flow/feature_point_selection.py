import numpy as np
import cv2


def feature_point_selection(
        frame: np.ndarray,
        quality_level: float = 0.3,
        quality_level_rv: float = 0.05,
        mask=None | np.ndarray,
        max_corners: int = 100,
        min_distance: int = 7,
) -> np.ndarray:
    """
    Extract feature points from the given frame
    :param frame:
    :param quality_level:
    :param quality_level_rv:
    :param mask:
    :param max_corners:
    :param min_distance:
    :return:
    """
    feature_params = {
        'maxCorners': max_corners,
        'qualityLevel': quality_level,
        'minDistance': min_distance
    }
    points = cv2.goodFeaturesToTrack(frame, mask=mask, **feature_params)

    while points is None:
        feature_params['qualityLevel'] = quality_level - quality_level_rv
        points = cv2.goodFeaturesToTrack(
            frame,
            mask=mask,
            **feature_params)

    return points


def special_feature_point_selection(
        frame: np.ndarray,
        quality_level: float = 0.3,
        quality_level_rv: float = 0.05,
        mask=None | np.ndarray,
        max_corners: int = 100,
        min_distance: int = 7,
        fpn: int = 5,
) -> np.ndarray:
    points = feature_point_selection(
        frame=frame,
        quality_level=quality_level,
        quality_level_rv=quality_level_rv,
        mask=mask,
        max_corners=max_corners,
        min_distance=min_distance,
    )

    if fpn > len(points):
        fpn = len(points)

    h = frame.shape[0] / 2
    w = frame.shape[1] / 2

    # TODO: Figure out how the top points are selected...
    p1 = points.copy()
    p1[:, :, 0] -= w
    p1[:, :, 1] -= h
    p1_1 = np.multiply(p1, p1)
    p1_2 = np.sum(p1_1, 2)
    p1_3 = np.sqrt(p1_2)
    p1_4 = p1_3[:, 0]
    p1_5 = np.argsort(p1_4)

    fp_map = np.zeros((fpn, 1, 2), dtype=np.float32)
    for inx in range(fpn):
        fp_map[inx, :, :] = points[p1_5[inx], :, :]

    return fp_map
