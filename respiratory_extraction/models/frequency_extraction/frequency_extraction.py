import numpy as np
from scipy.signal import find_peaks


# TODO: Clean up
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

        # Perform Fast Fourier Transform
        fft_result = np.fft.fft(self.data)

        # Calculate the frequencies corresponding to each FFT bin
        frequencies = np.fft.fftfreq(self.N, d=1 / self.sample_rate)

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

        distance = 60 / max_rr * self.sample_rate

        peaks, _ = find_peaks(
            self.data,
            height=height,
            threshold=threshold,
            distance=distance)

        return len(peaks) / self.Time

    def build_cross_curve(self) -> np.ndarray:
        """
        Build the cross curve
        :return:
        """

        shift_distance = int(self.sample_rate / 2)
        data_shift = np.zeros(self.data.shape) - 1
        data_shift[shift_distance:] = self.data[:-shift_distance]
        return self.data - data_shift

    def crossing_point(self) -> float:
        """
        Crossing Point Method
        :return:
        """

        cross_curve = self.build_cross_curve()

        zero_number = 0
        for inx in range(len(cross_curve) - 1):
            if cross_curve[inx] == 0:
                zero_number += 1
            elif cross_curve[inx] * cross_curve[inx + 1] < 0:
                zero_number += 1

        return (zero_number / 2) / self.Time

    def negative_feedback_crossover_point_method(
            self,
            quality_level=float(0.6)
    ) -> float:
        cross_curve = self.build_cross_curve()

        zero_number = 0
        zero_index = []
        for inx in range(len(cross_curve) - 1):
            if cross_curve[inx] == 0:
                zero_number += 1
                zero_index.append(inx)
            elif cross_curve[inx] * cross_curve[inx + 1] < 0:
                zero_number += 1
                zero_index.append(inx)

        rr_tmp = ((zero_number / 2) / self.Time)

        if len(zero_index) <= 1:
            return rr_tmp

        time_span = 60 / rr_tmp / 2 * self.sample_rate * quality_level
        zero_span = []
        for inx in range(len(zero_index) - 1):
            zero_span.append(zero_index[inx + 1] - zero_index[inx])

        while min(zero_span) < time_span:
            doubt_point = np.argmin(zero_span)
            zero_index.pop(doubt_point)
            zero_index.pop(doubt_point)
            if len(zero_index) <= 1:
                break
            zero_span = []
            for inx in range(len(zero_index) - 1):
                zero_span.append(zero_index[inx + 1] - zero_index[inx])

        return (zero_number / 2) / self.Time
