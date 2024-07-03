import numpy as np
import torch
import torchvision.transforms as transform


def preprocess(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert frames to torch.Tensor and apply normalization.
    :param frames: (N, H, W, C) numpy array
    :param device: torch.device
    :return: (N, C, H, W) torch.Tensor
    """
    # Permute from (N, H, W, C) to (N, C, H, W)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

    transforms = transform.Compose(
        [
            transform.ConvertImageDtype(torch.float32),
            transform.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        ]
    )
    return transforms(frames).to(device)
