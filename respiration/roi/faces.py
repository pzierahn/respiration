import cv2
import numpy as np
from typing import Sequence


def detect_faces(frame: np.ndarray, scale_factor: float = 1.3, min_neighbors: int = 5) -> Sequence[Sequence[int]]:
    """
    Detect faces in a frame using a Haar cascade classifier
    :param frame:
    :param scale_factor:
    :param min_neighbors:
    :return: list of detected faces (x, y, w, h)
    """

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade.detectMultiScale(frame, scale_factor, min_neighbors)
