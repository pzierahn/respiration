import cv2
import numpy as np

from respiration.roi import roi_to_mask


def find_feature_points(
        frame: np.ndarray,
        quality_level: float = 0.3,
        quality_level_rv: float = 0.05,
        roi: tuple[int, int, int, int] = None,
        max_corners: int = 100,
        min_distance: int = 7,
) -> np.ndarray:
    """
    Extract feature points from the given frame
    :param frame: The frame to extract feature points from
    :param quality_level: The quality level of the feature points
    :param quality_level_rv: The quality level of the feature points
    :param roi: The region of interest to extract feature points from (if None, extract feature points from the entire frame)
    :param max_corners: The maximum number of feature points to extract
    :param min_distance: The minimum distance between feature points
    :return: The extracted feature points
    """

    feature_params = {
        'maxCorners': max_corners,
        'qualityLevel': quality_level,
        'minDistance': min_distance
    }

    mask = None
    if roi is not None:
        mask = roi_to_mask(frame, roi)

    points = cv2.goodFeaturesToTrack(frame, mask=mask, **feature_params)

    while points is None:
        feature_params['qualityLevel'] = quality_level - quality_level_rv
        points = cv2.goodFeaturesToTrack(
            frame,
            mask=mask,
            **feature_params)

    return points


def select_feature_points(
        frame: np.ndarray,
        quality_level: float = 0.3,
        quality_level_rv: float = 0.05,
        fpn: int = None,
        roi: tuple[int, int, int, int] = None,
        max_corners: int = 100,
        min_distance: int = 7,
) -> np.ndarray:
    """
    Extract feature points from the given frame
    :param frame: The frame to extract feature points from
    :param quality_level: The quality level of the feature points
    :param quality_level_rv:
    :param fpn: The number of feature points to extract (if None, extract all feature points)
    :param roi: The region of interest to extract feature points from
    :param max_corners: The maximum number of feature points to extract
    :param min_distance: The minimum distance between feature points
    :return: The extracted feature points
    """

    points = find_feature_points(
        frame,
        quality_level=quality_level,
        quality_level_rv=quality_level_rv,
        roi=roi,
        max_corners=max_corners,
        min_distance=min_distance
    )

    # If the number of feature points is less than the required number, return all the points
    if fpn is None or len(points) < fpn:
        return points

    # Calculate the center of the roi or the frame
    if roi is None:
        w, h = frame.shape[1], frame.shape[0]
        center_x, center_y = int(w / 2), int(h / 2)
    else:
        x, y, w, h = roi
        center_x, center_y = int(x + w / 2), int(y + h / 2)

    # Calculate the distance of each point from the center
    points = points.reshape(-1, 2)
    distances = np.linalg.norm(points - np.array([center_x, center_y]), axis=1)

    # Sort the points based on the distance from the center
    sorted_points = points[np.argsort(distances)]

    # Return the first fpn points
    return sorted_points[:fpn].reshape(-1, 1, 2)
