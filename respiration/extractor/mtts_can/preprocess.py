import cv2
import numpy as np


def preprocess_video_frames(frames: np.ndarray, target_dim=(36, 36)) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses raw video frames by down sampling, normalizing (c(t + 1) − c(t))/(c(t + 1) + c(t)), and standardizing
    them.
    :param frames: Raw video frames
    :param target_dim: Target dimensions for resizing
    :return: Resized and standardized frames
    """

    # Resize all frames at once
    resized_frames = np.array([
        cv2.resize(frame, target_dim, interpolation=cv2.INTER_AREA) for frame in frames
    ])
    float_frames = resized_frames.astype(np.float32)

    # Calculate the nominator: c(t + 1) - c(t)
    nom = float_frames[1:] - float_frames[:-1]
    # Calculate the denominator: c(t + 1) + c(t)
    denom = float_frames[1:] + float_frames[:-1] + 1e-8

    # Normalize frames
    normalized_frames = nom / denom

    # Standardize frames using vectorized operations
    mean_val = np.mean(normalized_frames, axis=(0, 1, 2), keepdims=True)
    std_val = np.std(normalized_frames, axis=(0, 1, 2), keepdims=True)
    standardized_frames = (normalized_frames - mean_val) / (std_val + 1e-5)

    # Remove the first frame from the resized frames to match the number of frames
    resized_frames = resized_frames[1:]

    return resized_frames, standardized_frames
