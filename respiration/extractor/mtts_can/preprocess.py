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


def preprocess_frames_original(frames, dim=36):
    total_frames = len(frames)
    Xsub = np.zeros((total_frames, dim, dim, 3), dtype=np.float32)

    # Assuming all frames have the same dimensions
    height, width = frames[0].shape[:2]

    # Crop and resize each frame
    for i, img in enumerate(frames):
        float_img = img_as_float(img[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :])
        vidLxL = cv2.resize(
            float_img,
            (dim, dim),
            interpolation=cv2.INTER_AREA)
        # vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE)  # rotate 90 degree
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        Xsub[i, :, :, :] = vidLxL

    # Normalize frames for motion branch
    normalized_len = len(frames) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype=np.float32)
    for j in range(normalized_len - 1):
        # c(t + 1) - c(t) / c(t + 1) + c(t)
        dXsub[j, :, :, :] = (Xsub[j + 1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j + 1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)

    # Normalize raw frames for appearance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub / np.std(Xsub)
    Xsub = Xsub[:total_frames - 1, :, :, :]

    return Xsub, dXsub
