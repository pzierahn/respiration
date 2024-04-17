import numpy as np

from scipy.signal import find_peaks


# TODO: Clean up
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

    def fft(self, min_freq: float = 0, max_freq: float = float('inf')) -> float:
        """
        Calculate the respiratory frequency using the Fast Fourier Transform (FFT)
        :return:
        """

        """
        Extract the predominant frequency from the data using the Fast Fourier Transform.
        :param data: list of
        :param sampling_rate: frames per second
        :param min_freq: minimum frequency
        :param max_freq: maximum frequency
        :return: tuple of the fast fourier transform and frequency
        """

        # Perform Fast Fourier Transform
        fft_result = np.fft.fft(self.signal)

        # Calculate the frequencies corresponding to each FFT bin
        frequencies = np.fft.fftfreq(len(self.signal), d=1 / self.fs)

        # Filter frequencies in the range [min_freq, max_freq]
        frequency_filter = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies = frequencies[frequency_filter]
        fft_result = fft_result[frequency_filter]

        # Find the frequency corresponding to the maximum magnitude
        peak_freq = frequencies[np.argmax(np.abs(fft_result))]

        return float(peak_freq)

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
