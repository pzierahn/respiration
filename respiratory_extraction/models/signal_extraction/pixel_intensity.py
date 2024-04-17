import numpy as np


def average_pixel_intensity(frames: np.ndarray, roi: tuple[int, int, int, int] = None) -> np.ndarray:
    """
    Calculate the average pixel intensity in a region of interest (ROI) for each frame
    :param frames: numpy array of frames
    :param roi: region of interest (x, y, w, h)
    :return: average pixel value
    """

    if roi is None:
        roi = (0, 0, frames.shape[2], frames.shape[1])

    roi_x, roi_y, roi_w, roi_h = roi

    # Extract the region of interest from the frames
    roi_frames = frames[:, roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Calculate the average pixel value
    return roi_frames.mean(axis=(1, 2))
