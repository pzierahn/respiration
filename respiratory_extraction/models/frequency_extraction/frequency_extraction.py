from .fft import *
from .cross_point import *
from .peak_counting import *


class FrequencyExtraction:
    def __init__(self, data: np.ndarray, sample_rate: int):
        """
        Frequency Extraction Class
        :param data: Respiratory signal
        :param sample_rate: Sampling rate
        """

        self.data = data
        self.sample_rate = sample_rate
        self.N = len(data)
        self.Time = self.N / sample_rate

    def fft(self, min_freq: float = 0, max_freq: float = float('inf')) -> float:
        """
        Extract the predominant frequency from the data using the Fast Fourier Transform.
        :param min_freq: minimum frequency
        :param max_freq: maximum frequency
        :return: peak frequency
        """
        return frequency_from_fft(self.data, self.sample_rate, min_freq, max_freq)

    def peak_counting(self, height=None, threshold=None, max_rr=45) -> float:
        """
        Peak Counting Method
        :param height:
        :param threshold:
        :param max_rr:
        :return:
        """
        return frequency_from_peaks(self.data, self.sample_rate, height, threshold, max_rr)

    def crossing_point(self) -> float:
        """
        Crossing Point Method
        :return:
        """
        return frequency_from_crossing_point(self.data, self.sample_rate)

    def negative_feedback_crossover_point_method(
            self,
            quality_level=float(0.6)
    ) -> float:
        """
        Negative Feedback Crossover Point Method
        :param quality_level:
        :return:
        """
        return frequency_from_nfcp(
            self.data,
            self.sample_rate,
            quality_level=quality_level,
        )
