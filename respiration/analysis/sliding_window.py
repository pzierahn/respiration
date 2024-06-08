import numpy as np

from .psd import frequency_from_psd
from respiration.preprocessing import butterworth_filter


def sliding_window_psd(
        time_series: np.ndarray,
        sampling_rate: int,
        lowpass: float,
        highpass: float,
        window_size: int = 30,
        stride: int = 1) -> np.ndarray:
    """
    Calculate the frequency of the signal in a sliding window fashion.
    time_series: np.ndarray
        The input signal.
    sampling_rate: int
        The sampling rate of the signal in Hz.
    lowpass: float
        The lowpass cutoff frequency.
    highpass: float
        The highpass cutoff frequency.
    window_size: int
        The size of the window in seconds.
    stride: int
        The stride of the window in seconds.
    """

    time_series = butterworth_filter(
        time_series,
        sampling_rate,
        lowpass=lowpass,
        highpass=highpass)

    # Calculate the window size and stride in samples
    window_size *= sampling_rate
    stride *= sampling_rate

    results = []
    for inx in range(0, len(time_series) - window_size, stride):
        prediction_window = time_series[inx:inx + window_size]
        freq = frequency_from_psd(prediction_window, sampling_rate, lowpass, highpass)
        results.append(freq)

    return np.array(results)
