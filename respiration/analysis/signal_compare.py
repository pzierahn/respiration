from typing import Optional

from .fft import *
from .cross_point import *
from .peak_counting import *
from .distance import *

import respiration.preprocessing as preprocessing


class SignalCompare:
    """
    Class to compare two signals
    """
    prediction: np.ndarray
    ground_truth: np.ndarray
    sample_rate: int

    lowpass: Optional[float]
    highpass: Optional[float]

    def __init__(
            self,
            prediction: np.ndarray,
            prediction_sample_rate: int,
            ground_truth: np.ndarray,
            ground_truth_sample_rate: int,
            lowpass: Optional[float] = 0.08,
            highpass: Optional[float] = 0.6,
            detrend_tarvainen: bool = True,
            filter_signal: bool = True,
            normalize_signal: bool = True,
    ):
        if prediction_sample_rate > ground_truth_sample_rate:
            # Down-sample prediction
            prediction = preprocessing.resample_signal(prediction, len(ground_truth))
            sample_rate = ground_truth_sample_rate
        elif prediction_sample_rate < ground_truth_sample_rate:
            # Down-sample ground truth
            ground_truth = preprocessing.resample_signal(ground_truth, len(prediction))
            sample_rate = prediction_sample_rate
        else:
            sample_rate = prediction_sample_rate

        if detrend_tarvainen:
            prediction = preprocessing.detrend_tarvainen(prediction)
            ground_truth = preprocessing.detrend_tarvainen(ground_truth)

        if normalize_signal:
            prediction = preprocessing.normalize_signal(prediction)
            ground_truth = preprocessing.normalize_signal(ground_truth)

        if filter_signal:
            prediction = preprocessing.butterworth_filter(prediction, sample_rate, lowpass, highpass)
            ground_truth = preprocessing.butterworth_filter(ground_truth, sample_rate, lowpass, highpass)

        self.lowpass = lowpass
        self.highpass = highpass
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.sample_rate = sample_rate

    def compare_fft(self) -> tuple[float, float]:
        """Compare the frequency of the signals using the FFT method."""
        gt_frequency = frequency_from_fft(
            self.ground_truth,
            self.sample_rate,
            self.lowpass,
            self.highpass
        )
        pred_frequency = frequency_from_fft(
            self.prediction,
            self.sample_rate,
            self.lowpass,
            self.highpass
        )

        return gt_frequency, pred_frequency

    def compare_peaks(self) -> tuple[float, float]:
        """Compare the frequency of the signals using the peak counting method."""
        gt_frequency = frequency_from_peaks(self.ground_truth, self.sample_rate)
        pred_frequency = frequency_from_peaks(self.prediction, self.sample_rate)

        return gt_frequency, pred_frequency

    def compare_crossing_point(self) -> tuple[float, float]:
        """Compare the frequency of the signals using the crossing point method."""
        gt_frequency = frequency_from_crossing_point(self.ground_truth, self.sample_rate)
        pred_frequency = frequency_from_crossing_point(self.prediction, self.sample_rate)

        return gt_frequency, pred_frequency

    def compare_nfcp(self) -> tuple[float, float]:
        """Compare the frequency of the signals using the NFCP method."""
        gt_frequency = frequency_from_nfcp(self.ground_truth, self.sample_rate)
        pred_frequency = frequency_from_nfcp(self.prediction, self.sample_rate)

        return gt_frequency, pred_frequency

    def pearson_correlation(self) -> float:
        """Calculate the Pearson correlation between the two signals."""
        return pearson_correlation(self.ground_truth, self.prediction)

    def distance_mse(self) -> float:
        """Calculate the mean absolute error between the two signals."""
        return distance_mse(self.ground_truth, self.prediction)
