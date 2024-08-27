import os

from typing import Optional
from keras.models import Model
from respiration.extractor.mtts_can.models.mtts_can import mtts_can


def calculate_cutoff(size: int, frame_depth: int) -> int:
    """The model expects a number of frames that is a multiple of frame_depth"""
    return (size // frame_depth) * frame_depth


def load_model(
        model_checkpoint: Optional[str] = None,
        frame_depth: int = 10,
        nb_filters1: int = 32,
        nb_filters2: int = 64,
        input_shape: tuple[int, int, int] = (36, 36, 3),
) -> Model:
    """Load the pre-trained model"""
    if model_checkpoint is None:
        model_checkpoint = os.path.join('..', '..', 'data', 'mtts_can', 'mtts_can.hdf5')

    model = mtts_can(frame_depth, nb_filters1, nb_filters2, input_shape)
    model.load_weights(model_checkpoint)

    return model
