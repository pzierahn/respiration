import numpy as np

from .chest import chest_from_face
from .faces import detect_faces
from .yolo import YOLO

_yolo = None


def get_roi_areas(frame: np.ndarray) -> list[tuple[list[int], str]]:
    """
    Get full, chest and person regions of interest (ROIs) for the given frame.
    :param frame: The frame to get the ROIs from
    :return: A list of tuples containing the ROI and the name of the ROI
    """

    global _yolo

    if _yolo is None:
        # Load the YOLO model
        _yolo = YOLO()

    regions = [
        # ROI for the full frame
        ((0, 0, frame.shape[1], frame.shape[0]), 'full')
    ]

    # Calculate the region of interest (ROI) based on the face
    faces = detect_faces(frame)
    if len(faces) == 1:
        chest_roi = chest_from_face(faces[0])
        regions.append((chest_roi, 'chest'))

    # Use the detected person to create a mask
    persons = _yolo.detect_classes(frame, clazz='person')
    if len(persons) == 1:
        regions.append((persons[0], 'person'))

    return regions
