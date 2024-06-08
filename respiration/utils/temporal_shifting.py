import cv2
import numpy as np
from skimage.util import img_as_float


def preprocess_video_frames(frames: np.ndarray, dim=36) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses raw video frames by down sampling, normalizing (c(t + 1) − c(t))/(c(t + 1) + c(t)), and standardizing
    them.
    :param frames: Raw video frames
    :param dim: Target dimensions for resizing
    :return: Resized and standardized frames
    """
    total_frames = len(frames)
    resized = np.zeros((total_frames, dim, dim, 3), dtype=np.float32)

    # Crop and resize each frame
    for inx, img in enumerate(frames):
        float_img = img_as_float(img)
        resized_img = cv2.resize(
            float_img,
            (dim, dim),
            interpolation=cv2.INTER_AREA)
        resized_img = resized_img.astype('float32')
        resized_img[resized_img > 1] = 1
        resized_img[resized_img < (1 / 255)] = 1 / 255
        resized[inx, :, :, :] = resized_img

    # Normalize frames for motion branch
    normalized = ((resized[1:] - resized[:-1]) /
                  (resized[1:] + resized[:-1]))
    normalized = normalized / np.std(normalized)

    resized = resized - np.mean(resized)
    resized = resized / np.std(resized)
    resized = resized[:total_frames - 1, :, :, :]

    return resized, normalized
