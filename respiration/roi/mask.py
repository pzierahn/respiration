import numpy as np


def roi_to_mask(
        frame: np.ndarray,
        roi: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Create a mask for the region of interest coordinates (x, y, w, h)
    :param frame:
    :param roi:
    :return:
    """
    x, y, w, h = roi
    roi_mask = np.zeros_like(frame)
    roi_mask[y:y + h, x:x + w] = 255
    return roi_mask
