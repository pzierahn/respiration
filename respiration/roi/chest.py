import numpy as np
from typing import Sequence
from .faces import detect_faces


def chest_from_face(face: Sequence[int],
                    scale_w: float = 0,
                    scale_h: float = 0) -> tuple[int, int, int, int]:
    """
    Calculate the region of interest (ROI) in the chest area based on the face
    :param face: coordinates of the face (x, y, w, h)
    :param scale_w: scale factor for the ROI width
    :param scale_h: scale factor for the ROI height
    :return: tuple of the chest region (x, y, w, h)
    """

    x, y, w, h = face

    delta_x = int(w * scale_w)
    delta_y = int(h * scale_h)

    # Calculate the region of interest (ROI) based on the face
    chest_x = x - delta_x
    chest_y = int(y + h + h * 0.7) - delta_y
    chest_w = w + delta_x * 2
    chest_h = int(h * 0.5) + delta_y

    return chest_x, chest_y, chest_w, chest_h


def detect_chest(
        frame: np.ndarray,
        scale_w: float = 0,
        scale_h: float = 0,
        face_scale_factor: float = 1.3,
        face_min_neighbors: int = 5,
) -> tuple[int, int, int, int]:
    """
    Calculate the chest region of interest (ROI) based on the face detected in frame
    :param frame: The frame to detect the chest from
    :param scale_w: scale factor for the ROI width
    :param scale_h: scale factor for the ROI height
    :param face_scale_factor: scaleFactor parameter for detectMultiScale
    :param face_min_neighbors: minNeighbors parameter for detectMultiScale
    :return: tuple of the chest region (x, y, w, h)
    """

    faces = detect_faces(frame, face_scale_factor, face_min_neighbors)

    assert len(faces) == 1, f"Expected 1 face, but got {len(faces)}"

    return chest_from_face(faces[0], scale_w, scale_h)
