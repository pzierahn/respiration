import cv2
import numpy as np


def track_feature_point_movement(
        frames: np.ndarray,
        feature_points: np.ndarray,
        win_size: tuple[int, int] = (15, 15),
        max_level: int = 2) -> np.ndarray:
    """
    Extract the movement of the feature points in the frames using the Lucas-Kanade optical flow method.
    :param frames: The frames to extract the feature points from.
    :param feature_points: The feature points to track.
    :param win_size: The window size of the optical flow.
    :param max_level: The maximum level of the pyramid for the optical flow.
    :return: The movement of the feature points in the frames. (N, F, 2)
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


def calculate_feature_point_amplitudes(motion_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the amplitudes of the feature points from the motion matrix.
    :param motion_matrix: The motion matrix of the feature points. (N, F, 2)
    :return: The amplitudes of the feature points. (N, F)
    """

    # Calculate the amplitudes of the feature points
    return np.sqrt(motion_matrix[:, :, 0] ** 2 + motion_matrix[:, :, 1] ** 2)
