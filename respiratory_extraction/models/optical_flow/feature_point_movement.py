import cv2
import numpy as np


def extract_feature_point_movement(
        frames: np.ndarray,
        feature_points: np.ndarray,
        win_size: tuple[int, int] = (15, 15),
        max_level: int = 2) -> np.ndarray:
    """
    Extract the movement of the feature points in the frames using the Lucas-Kanade optical flow method.
    :param frames:
    :param feature_points:
    :param win_size:
    :param max_level:
    :return:
    """

    lk_params = {
        'winSize': win_size,
        'maxLevel': max_level,
    }
    total_frame = len(frames)

    # Store the feature points for each frame
    feature_point_matrix = np.zeros((int(total_frame), feature_points.shape[0], 2))

    # Store the feature points for the first frame
    feature_point_matrix[0, :, 0] = feature_points[:, 0, 0].T
    feature_point_matrix[0, :, 1] = feature_points[:, 0, 1].T

    # Calculate the optical flow of the feature points for each frame
    for inx in range(1, total_frame):
        current_frame = frames[inx - 1]
        next_frame = frames[inx]

        new_positions, _, _ = cv2.calcOpticalFlowPyrLK(
            current_frame,
            next_frame,
            feature_points,
            None,
            **lk_params)

        feature_points = new_positions.reshape(-1, 1, 2)
        feature_point_matrix[inx, :, 0] = feature_points[:, 0, 0].T
        feature_point_matrix[inx, :, 1] = feature_points[:, 0, 1].T

    return feature_point_matrix
