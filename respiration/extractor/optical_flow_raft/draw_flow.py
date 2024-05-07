import cv2
import torch
import numpy as np
from torchvision.utils import flow_to_image


def draw_flow(frame: np.ndarray, flow: np.ndarray, step: int = 20):
    """
    Plots the optical flow vectors on the image.
    :param frame: The frame on which to plot the optical flow vectors. (H, W, C)
    :param flow: The optical flow vectors. (2, H, W)
    :param step: The step size for the grid.
    :return: The frame with the optical flow vectors plotted on it.
    """

    h, w = frame.shape[:2]
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[:, y, x]

    visualization = frame.copy()

    # Draw arrows
    for (x0, y0, dx, dy) in zip(x, y, fx, fy):
        # Length of the arrow is sqrt(dx^2 + dy^2)
        # length = np.sqrt(dx ** 2 + dy ** 2)
        # if length > 20:
        #     continue

        end_point = (int(x0 + dx), int(y0 + dy))
        cv2.arrowedLine(
            visualization,
            (x0, y0),
            end_point,
            color=(255, 0, 0),
            thickness=1,
            tipLength=0.25,
        )

    return visualization


def image_from_flow(flow: np.ndarray) -> np.ndarray:
    """
    Converts the optical flow vectors to an image of magnitudes.
    :param flow: The optical flow vectors. (2, H, W)
    :return: The image of the optical flow vectors. (H, W, C)
    """
    input = torch.from_numpy(flow)
    flow_image = flow_to_image(input)
    return flow_image.numpy().transpose(1, 2, 0)
