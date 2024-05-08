from typing import Optional

import cv2
import numpy as np

import torch


def preprocess_frames(frames: np.ndarray, big_res: int = 144, small_res: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the frames to be fed into the model. This will produce the Big Slow Branch and the Small Fast Branch.
    :param frames: The frames to be preprocessed (N, H, W, C)
    :param big_res: The resolution of the big branch frames
    :param small_res: The resolution of the small branch frames
    :return: The preprocessed frames for the Big Slow Branch and the Small Fast Branch
    """

    # Center crop frames to square shape
    h, w, _ = frames[0].shape
    crop_size = min(h, w)
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    frames = [frame[start_y:start_y + crop_size, start_x:start_x + crop_size] for frame in frames]

    # Convert frames to floating point
    frames = np.array(frames, dtype=np.float32)

    # Generate Small branch inputs (normalized difference frames)
    diff_frames = frames[1:] - frames[:-1]
    sum_frames = frames[1:] + frames[:-1]
    small_inputs = diff_frames / (sum_frames + 1e-7)
    small_inputs = (small_inputs - np.mean(small_inputs)) / np.std(small_inputs)
    small_inputs = [cv2.resize(frame, (small_res, small_res)) for frame in small_inputs]

    # Add a zero frame at the beginning to match the number of frames
    small_inputs = [np.zeros_like(small_inputs[0])] + small_inputs
    small_inputs = np.array(small_inputs)

    # Replace the nan values with zeros
    small_inputs = np.nan_to_num(small_inputs)

    # Generate Big branch inputs (raw frames)
    big_inputs = (frames - np.mean(frames)) / np.std(frames)
    big_inputs = np.array([cv2.resize(frame, (big_res, big_res)) for frame in big_inputs])

    return big_inputs, small_inputs


def convert_to_input(
        big: np.ndarray,
        small: np.ndarray,
        device: Optional[torch.device] = None) -> (torch.Tensor, torch.Tensor):
    """
    Convert the frames to a tensor and transform them to the shape expected by the model (N, C, H, W)
    :param big: The preprocessed frames for the Big Slow Branch
    :param small: The preprocessed frames for the Small Fast Branch
    :param device: The device to use for the tensor
    :return: The input tensors for the BigSmallModel
    """

    # Convert the frames to a tensor
    big_tensor = torch.tensor(big, device=device)
    small_tensor = torch.tensor(small, device=device)

    # Transform the tensor to the shape expected by the model (N, C, H, W)
    big_tensor = big_tensor.permute(0, 3, 1, 2)
    small_tensor = small_tensor.permute(0, 3, 1, 2)

    return big_tensor, small_tensor
