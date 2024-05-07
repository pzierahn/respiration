import numpy as np
import torch
import torchvision.transforms as transform


def preprocess(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

    transforms = transform.Compose(
        [
            transform.ConvertImageDtype(torch.float32),
            transform.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        ]
    )
    return transforms(frames).to(device)
