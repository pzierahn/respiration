import numpy as np


def peak_frequency(data: list[int] | np.ndarray,
                   sampling_rate: int,
                   min_freq: float = 0.0,
                   max_freq: float = float('inf')) -> float:
    """
    Extract the predominant frequency from the data using the Fast Fourier Transform.
    :param data: list of
    :param sampling_rate: frames per second
    :param min_freq: minimum frequency
    :param max_freq: maximum frequency
    :return: tuple of the fast fourier transform and frequency
    """

    # Perform Fast Fourier Transform
    fft_result = np.fft.fft(data)

    # Calculate the frequencies corresponding to each FFT bin
    frequencies = np.fft.fftfreq(len(data), d=1 / sampling_rate)

    # Filter frequencies in the range [min_freq, max_freq]
    frequency_filter = (frequencies >= min_freq) & (frequencies <= max_freq)
    frequencies = frequencies[frequency_filter]
    fft_result = fft_result[frequency_filter]

    # Find the frequency corresponding to the maximum magnitude
    peak_freq = frequencies[np.argmax(np.abs(fft_result))]

    return float(peak_freq)