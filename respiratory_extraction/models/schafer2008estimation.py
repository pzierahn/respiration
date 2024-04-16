import numpy as np
from numpy.fft import fft, fftfreq


def acf_adv(pixel_values: list[int],
            fps: int,
            min_freq: float = 0,
            max_freq: float = float('inf')) -> float:
    """
    Estimates breathing rate from pixel_values intervals using the Autocorrelation Advanced Method. This method is from
    the paper "Estimation of breathing rate from respiratory sinus arrhythmia: comparison of various methods" by Axel
    Schäfer and Karl W Kratky.

    Source: https://github.com/arturomoncadatorres/breathing-rate-rsa
    :param pixel_values: list of pixel values
    :param fps: frames per second
    :param min_freq: minimum frequency
    :param max_freq: maximum frequency
    :return: breathing rate
    """

    # Calculate the differences of subsequent intervals (DNN)
    dnn = np.diff(pixel_values)

    # Calculate the ACF of DNN
    acf = np.correlate(dnn, dnn, mode='full')[len(dnn) - 1:]
    acf /= acf[0]  # Normalize

    # Calculate the power spectrum of the ACF
    # Use the provided sampling rate for accurate frequency calculation
    freqs = fftfreq(len(acf), d=1 / fps)
    power_spectrum = np.abs(fft(acf)) ** 2

    # Find relevant frequencies within the respiratory band (0.1 - 0.5 Hz)
    respiratory_band = (freqs >= min_freq) & (freqs <= max_freq)
    relevant_frequencies = freqs[respiratory_band]
    relevant_powers = power_spectrum[respiratory_band]

    # Calculate the median power within the respiratory band
    median_power = np.median(relevant_powers)

    # Select frequencies with power above the median
    selected_freqs = relevant_frequencies[relevant_powers > median_power]
    selected_powers = relevant_powers[relevant_powers > median_power]

    # Calculate the weighted mean of selected frequencies
    breathing_freq = np.sum(selected_freqs * selected_powers) / np.sum(selected_powers)

    return breathing_freq
