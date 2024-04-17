from typing import Optional

from .fft import *
from .cross_point import *
from .peak_counting import *

import respiratory_extraction.models.signal_preprocessing as preprocessing


class FrequencyExtractor:
    data: np.ndarray
    sample_rate: int

    lowpass: Optional[float]
    highpass: Optional[float]

    def __init__(
            self,
            data: np.ndarray,
            sample_rate: int,
            lowpass: Optional[float] = 0.0,
            highpass: Optional[float] = float('inf'),
            filter_signal: Optional[bool] = True,
            normalize_signal: Optional[bool] = True,
    ):
        if filter_signal:
            data = preprocessing.butterworth_filter(data, sample_rate, lowpass, highpass)

        if normalize_signal:
            data = preprocessing.normalize_signal(data)

        self.data = data
        self.sample_rate = sample_rate
        self.lowpass = lowpass
        self.highpass = highpass

    def frequency_from_fft(self, min_freq: Optional[float], max_freq: Optional[float]) -> float:
        """
        Extract the predominant frequency from the data using the Fast Fourier Transform.
        :param min_freq: minimum frequency
        :param max_freq: maximum frequency
        :return: peak frequency
        """

        if min_freq is None:
            min_freq = self.lowpass

        if max_freq is None:
            max_freq = self.highpass

        return frequency_from_fft(self.data, self.sample_rate, min_freq, max_freq)

    def frequency_from_peaks(self, height=None, threshold=None, max_rr=45) -> float:
        """
        Extract the predominant frequency from the data using the peak counting method.
        :param height:
        :param threshold:
        :param max_rr:
        :return:
        """
        return frequency_from_peaks(self.data, self.sample_rate, height, threshold, max_rr)

    def frequency_from_crossing_point(self) -> float:
        """
        Extract the predominant frequency from the data using the crossing point method.
        :return:
        """
        return frequency_from_crossing_point(self.data, self.sample_rate)

    def frequency_from_nfcp(
            self,
            quality_level: float = 0.6
    ) -> float:
        """
        Extract the predominant frequency from the data using the negative feedback crossover point method.
        :param quality_level:
        :return:
        """
        return frequency_from_nfcp(
            self.data,
            self.sample_rate,
            quality_level=quality_level,
        )
