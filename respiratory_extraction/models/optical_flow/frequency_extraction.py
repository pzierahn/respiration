import numpy as np

from scipy.fftpack import fft
from scipy.signal import find_peaks


class FrequencyExtraction:
    def __init__(self, respiratory_signal: np.ndarray, fs: float):
        """
        Frequency Extraction Class
        :param respiratory_signal: Respiratory signal
        :param fs: Sampling frequency
        """
        self.signal = respiratory_signal
        self.fs = fs
        self.N = len(respiratory_signal)
        self.Time = self.N / fs

    def fft(self) -> float:
        """
        Calculate the respiratory frequency using the Fast Fourier Transform (FFT)
        :return:
        """

        fft_y = fft(self.signal)
        abs_y = np.abs(fft_y)
        normalization_y = abs_y / self.N
        normalization_half_y = normalization_y[range(int(self.N / 2))]
        sorted_indices = np.argsort(normalization_half_y)

        max_frequency = self.fs
        f = np.linspace(0, max_frequency, self.N)
        return f[sorted_indices[-2]]

    def peak_counting(self, height=None, threshold=None, max_rr=45) -> float:
        """
        Peak Counting Method
        :param height:
        :param threshold:
        :param max_rr:
        :return:
        """

        distance = 60 / max_rr * self.fs

        peaks, _ = find_peaks(
            self.signal,
            height=height,
            threshold=threshold,
            distance=distance)

        return len(peaks) / self.Time

    def _get_cross_curve(self) -> np.ndarray:
        shift_distance = int(self.fs / 2)
        data_shift = np.zeros(self.signal.shape) - 1
        data_shift[shift_distance:] = self.signal[:-shift_distance]
        return self.signal - data_shift

    def crossing_point(self) -> float:
        """
        Crossing Point Method
        :return:
        """

        cross_curve = self._get_cross_curve()

        zero_number = 0
        zero_index = []
        for inx in range(len(cross_curve) - 1):
            if cross_curve[inx] == 0:
                zero_number += 1
                zero_index.append(inx)
            else:
                if cross_curve[inx] * cross_curve[inx + 1] < 0:
                    zero_number += 1
                    zero_index.append(inx)

        return (zero_number / 2) / (self.N / self.fs)

    def negative_feedback_crossover_point_method(
            self,
            quality_level=float(0.6)
    ) -> float:
        cross_curve = self._get_cross_curve()

        zero_number = 0
        zero_index = []
        for i in range(len(cross_curve) - 1):
            if cross_curve[i] == 0:
                zero_number += 1
                zero_index.append(i)
            else:
                if cross_curve[i] * cross_curve[i + 1] < 0:
                    zero_number += 1
                    zero_index.append(i)

        rr_tmp = ((zero_number / 2) / (self.N / self.fs))

        if len(zero_index) <= 1:
            return rr_tmp

        time_span = 60 / rr_tmp / 2 * self.fs * quality_level
        zero_span = []
        for i in range(len(zero_index) - 1):
            zero_span.append(zero_index[i + 1] - zero_index[i])

        while min(zero_span) < time_span:
            doubt_point = np.argmin(zero_span)
            zero_index.pop(doubt_point)
            zero_index.pop(doubt_point)
            if len(zero_index) <= 1:
                break
            zero_span = []
            for i in range(len(zero_index) - 1):
                zero_span.append(zero_index[i + 1] - zero_index[i])

        return (zero_number / 2) / (self.N / self.fs)
