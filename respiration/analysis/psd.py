import numpy as np
from scipy.signal import periodogram


def filtered_periodogram(
        time_series: np.ndarray,
        sampling_rate: int,
        min_freq: float = 0,
        max_freq: float = float('inf')) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the power spectral density (PSD) of a signal within a given frequency range.
    :param time_series: Respiratory signal
    :param sampling_rate: Sampling rate
    :param min_freq: minimum frequency
    :param max_freq: maximum frequency
    :return: Frequencies and FFT result
    """

    # Compute the power spectral density (PSD) using periodogram
    freq, psd = periodogram(time_series, fs=sampling_rate)

    # Find the indices corresponding to the frequency range
    idx = (freq >= min_freq) & (freq <= max_freq)

    # Extract the frequencies and PSDs within the specified range
    freq_range = freq[idx]
    psd_range = psd[idx]

    return freq_range, psd_range


def frequency_from_psd(
        time_series: np.ndarray,
        sampling_rate: int,
        min_freq: float = 0,
        max_freq: float = float('inf')) -> float:
    """
    Finds the frequency with the maximum power within a given frequency range.
    :param time_series: Respiratory signal
    :param sampling_rate: Sampling rate
    :param min_freq: minimum frequency
    :param max_freq: maximum frequency
    :return: Frequency with the maximum power
    """

    # Compute the power spectral density (PSD) using periodogram
    freq, psd = filtered_periodogram(time_series, sampling_rate, min_freq, max_freq)

    # Find the index of the maximum PSD within the range
    max_idx = np.argmax(psd)

    # Get the frequency corresponding to the maximum PSD
    return float(freq[max_idx])
