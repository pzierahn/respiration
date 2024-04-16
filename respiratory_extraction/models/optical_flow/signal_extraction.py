from .pre_processing import *
from .feature_point_selection import *
from .feature_point_movement import *


def extract_respiratory_signal(
        frames: np.ndarray,
        fps: int,
        fpn: None | int,
        roi_mask: None | np.ndarray = None,
        quality_level: float = 0.3,
        quality_level_rv: float = 0.05,
        lowpass: float = 0.1,
        highpass: float = 0.6,
        use_cgof: bool = False,
        use_filter: bool = False,
        use_normalization: bool = False,
) -> np.ndarray:
    feature_points = get_feature_points(
        frames[0],
        fpn=fpn,
        quality_level=quality_level,
        quality_level_rv=quality_level_rv,
        roi_mask=roi_mask,
    )

    # Extract the movement of the feature points for each frame
    feature_point_movements = extract_feature_point_movement(frames, feature_points)

    # Calculate the amplitude of the feature points for each frame
    point_amplitudes = np.sqrt(feature_point_movements[:, :, 0] ** 2 + feature_point_movements[:, :, 1] ** 2)

    # Calculate the amplitude of the feature points for each frame
    respiratory_signal = np.sum(point_amplitudes, 1) / point_amplitudes.shape[1]

    # Correlation-Guided Optical Flow Method
    if use_cgof:
        respiratory_signal = correlation_guided_optical_flow_method(point_amplitudes, respiratory_signal)

    # Butterworth Filter
    if use_filter:
        respiratory_signal = butterworth_filter(
            respiratory_signal,
            fps,
            lowpass=lowpass,
            highpass=highpass,
        )

    # Normalization
    if use_normalization:
        respiratory_signal = normalize_signal(respiratory_signal)

    return respiratory_signal
