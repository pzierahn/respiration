import os
from typing import Optional

from respiration.extractor.mtts_can.models.mtts_can import mtts_can


def load_model(
        model_checkpoint: Optional[str] = None,
        frame_depth: int = 10,
        nb_filters1: int = 32,
        nb_filters2: int = 64,
        input_shape: tuple[int, int, int] = (36, 36, 3),
):
    if model_checkpoint is None:
        model_checkpoint = os.path.join('..', 'data', 'mtts_can', 'mtts_can.hdf5')

    model = mtts_can(frame_depth, nb_filters1, nb_filters2, input_shape)
    model.load_weights(model_checkpoint)

    return model
