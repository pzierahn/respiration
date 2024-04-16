import numpy as np
from scipy.fft import fft, fftfreq


def average_pixel_intensity(frames: np.ndarray, roi=None) -> list[int]:
    """
    Calculate the average pixel intensity in a region of interest (ROI) for each frame
    :param frames: numpy array of frames
    :param roi: region of interest (x, y, w, h)
    :return: average pixel value
    """

    if roi is None:
        roi = (0, 0, frames.shape[2], frames.shape[1])

    roi_x, roi_y, roi_w, roi_h = roi

    # Extract the region of interest from the frames
    roi_frames = frames[:, roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Calculate the average pixel value
    return roi_frames.mean(axis=(1, 2))


def calculate_fft(pixel_values: list[int],
                  fps: int,
                  min_freq=float(0),
                  max_freq=float('inf')) -> tuple[np.array, np.array]:
    """
    Calculate the frequency of the fast fourier transform of the pixel values. The negative frequencies are removed. The
    frequency is also limited to the range between minFreq and maxFreq.
    :param pixel_values: list of pixel values
    :param fps: frames per second
    :param min_freq: minimum frequency
    :param max_freq: maximum frequency
    :return: tuple of the fast fourier transform and frequency
    """

    # Calculate the fast fourier transform of the thorax abdomen data
    pixels_fft = fft(pixel_values)

    # Calculate the frequency
    freq = np.fft.fftfreq(len(pixels_fft), 1 / fps)

    # Remove the negative frequencies
    pixels_fft = pixels_fft[freq > 0]
    freq = freq[freq > 0]

    # Limit the frequency to the range between minFreq and maxFreq
    filter_freq = (freq >= min_freq) & (freq <= max_freq)
    pixels_fft = pixels_fft[filter_freq]
    freq = freq[filter_freq]

    return pixels_fft, freq


def calculate_respiratory_rate(pixels_fft: np.array, freq: np.array) -> tuple[float, float]:
    """
    Calculate the respiratory rate from the fast fourier transform of the pixel values
    :param pixels_fft: fast fourier transform of the pixel values
    :param freq: frequency of the fast fourier transform
    :return: peak frequency and respiratory rate
    """

    # Find the peak frequency
    peak_freq = freq[np.argmax(np.abs(pixels_fft))]

    # Calculate the respiratory rate
    respiratory_rate = peak_freq * 60

    return peak_freq, respiratory_rate
