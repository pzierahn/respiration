import numpy as np
from typing import Optional


def average_pixel_intensity(frames: np.ndarray, roi: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Calculate the average pixel intensity in a region of interest (ROI) for each frame. This function is used for
    grayscale frames.
    :param frames: numpy array of frames
    :param roi: region of interest (x, y, w, h)
    :return: average pixel value
    """

    if roi is None:
        # Default to the entire frame
        roi = (0, 0, frames.shape[2], frames.shape[1])

    roi_x, roi_y, roi_w, roi_h = roi

    # Extract the region of interest from the frames
    roi_frames = frames[:, roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Calculate the average pixel value
    return roi_frames.mean(axis=(1, 2))


def average_pixel_intensity_rgb(frames: np.ndarray, roi: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Calculate the average pixel intensity in a region of interest (ROI) for each frame. This function is used for RGB
    frames.
    :param frames: numpy array of frames
    :param roi: region of interest (x, y, w, h)
    :return: average pixel value
    """

    # For each channel, calculate the average pixel intensity
    channels = []
    for idx in range(frames.shape[3]):
        channel = average_pixel_intensity(frames[:, :, :, idx], roi)
        channels.append(channel)

    # Average the pixel values across the channels
    stack = np.stack(channels, axis=1)
    return stack.mean(axis=1)
