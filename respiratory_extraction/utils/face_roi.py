import cv2
from typing import Sequence


def detect_faces(frame, scale_factor=float(1.3), min_neighbors=int(5)) -> Sequence[Sequence[int]]:
    """
    Detect faces in a frame using a Haar cascade classifier
    :param frame:
    :param scale_factor:
    :param min_neighbors:
    :return: list of detected faces (x, y, w, h)
    """

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade.detectMultiScale(frame, scale_factor, min_neighbors)


def roi_from_face(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    """
    Calculate the region of interest (ROI) in the chest area based on the face
    :param x: x-coordinate of the face
    :param y: y-coordinate of the face
    :param w: width of the face
    :param h: height of the face
    :return: tuple of the chest region (x, y, w, h)
    """

    # This calculation is based on Cheng_Liu_2023
    chest_x = x
    chest_y = int(y + h + h * 0.7)
    chest_w = w
    chest_h = int(h * 0.5)

    return chest_x, chest_y, chest_w, chest_h
