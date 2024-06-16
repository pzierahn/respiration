import torch
import numpy as np
from torchvision import transforms


def resize_and_center_frames(frames: np.ndarray, dim: int):
    """
    Resize and center crop the frames to the new dimension.
    :param frames: np.ndarray of shape (T, H, W, C)
    :param dim: int, new dimension to resize the frames
    :return: torch.Tensor of shape (T, C, dim, dim)
    """
    preprocess = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        # Center Crop the frames
        transforms.CenterCrop(dim),
        transforms.ToTensor()
    ])

    return torch.stack([preprocess(frame) for frame in frames], dim=0)
