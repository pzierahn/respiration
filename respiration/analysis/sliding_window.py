import numpy as np

from typing import Optional

from .cross_point import (
    frequency_from_crossing_point,
    frequency_from_nfcp,
)
from .distance import (
    pearson_correlation,
    spearman_correlation,
)
from .peak_counting import frequency_from_peaks
from .psd import frequency_from_psd
from .preprocessing import *


def sliding_window_analysis(
        time_series: np.ndarray,
        sampling_rate: int,
        lowpass: float,
        highpass: float,
        window_size: int = 30,
        stride: int = 1) -> dict[str, np.ndarray]:
    """
    Calculate the frequency of the signal in a sliding window fashion.
    time_series: np.ndarray
        The input signal.
    sampling_rate: int
        The sampling rate of the signal in Hz.
    lowpass: float
        The lowpass cutoff frequency.
    highpass: float
        The highpass cutoff frequency.
    window_size: int
        The size of the window in seconds.
    stride: int
        The stride of the window in seconds.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary with the following keys:
        - cp: The frequency calculated using the crossing point method.
        - nfcp: The frequency calculated using the negative first crossing point method.
        - pk: The frequency calculated using the peak counting method.
        - psd: The frequency calculated using the power spectral density method.
    """

    time_series = butterworth_filter(
        time_series,
        sampling_rate,
        lowpass=lowpass,
        highpass=highpass)

    # Calculate the window size and stride in samples
    window_size *= sampling_rate
    stride *= sampling_rate

    results = {
        'cp': [],
        'nfcp': [],
        'pk': [],
        'psd': []
    }

    for inx in range(0, len(time_series) - window_size, stride):
        prediction_window = time_series[inx:inx + window_size]

        frequency_cp = frequency_from_crossing_point(prediction_window, sampling_rate)
        frequency_nfcp = frequency_from_nfcp(prediction_window, sampling_rate)
        frequency_pk = frequency_from_peaks(prediction_window, sampling_rate)
        frequency_psd = frequency_from_psd(prediction_window, sampling_rate, lowpass, highpass)

        results['cp'].append(frequency_cp)
        results['nfcp'].append(frequency_nfcp)
        results['pk'].append(frequency_pk)
        results['psd'].append(frequency_psd)

    # Convert the results to a numpy array
    results = {key: np.array(value) for key, value in results.items()}

    return results


class Analysis:
    lowpass: Optional[float]
    highpass: Optional[float]

    detrend: bool
    normalize: bool
    filter_signal: bool

    window_size: int
    stride: int

    # Analysis results
    prediction_results: dict[str, np.ndarray]
    ground_truth_results: dict[str, np.ndarray]

    def __init__(
            self,
            lowpass: Optional[float] = 0.08,
            highpass: Optional[float] = 0.6,
            detrend: bool = False,
            normalize: bool = True,
            filter_signal: bool = True,
            window_size: int = 30,
            stride: int = 1
    ):
        self.lowpass = lowpass
        self.highpass = highpass

        self.detrend = detrend
        self.normalize = normalize
        self.filter_signal = filter_signal

        self.window_size = window_size
        self.stride = stride

        self.prediction_results = {
            'cp': np.array([]),
            'nfcp': np.array([]),
            'pk': np.array([]),
            'psd': np.array([]),
        }

        self.ground_truth_results = {
            'cp': np.array([]),
            'nfcp': np.array([]),
            'pk': np.array([]),
            'psd': np.array([]),
        }

    def __preprocess(
            self, prediction: np.ndarray,
            ground_truth: np.ndarray,
            sample_rate: int) -> tuple[np.ndarray, np.ndarray]:

        assert prediction.shape == ground_truth.shape, \
            (f'Prediction and ground truth signals must have the same shape. Got prediction shape: {prediction.shape}, '
             f'ground truth shape: {ground_truth.shape}')

        if self.detrend:
            prediction = detrend_tarvainen(prediction)
            ground_truth = detrend_tarvainen(ground_truth)

        if self.filter_signal:
            prediction = butterworth_filter(
                prediction,
                sample_rate,
                self.lowpass,
                self.highpass)
            ground_truth = butterworth_filter(
                ground_truth,
                sample_rate,
                self.lowpass,
                self.highpass)

        if self.normalize:
            prediction = normalize_signal(prediction)
            ground_truth = normalize_signal(ground_truth)

        return prediction, ground_truth

    def add_data(self, prediction: np.ndarray, ground_truth: np.ndarray, sample_rate: int):
        """
        Add data to the analysis.
        :param prediction: The predicted signal.
        :param ground_truth: The ground truth signal.
        :param sample_rate: The sampling rate of the signals in Hz.
        """
        prediction, ground_truth = self.__preprocess(prediction, ground_truth, sample_rate)

        # Calculate the window size and stride in samples
        window_size = self.window_size * sample_rate
        stride = self.stride * sample_rate

        metrics = {
            'cp': frequency_from_crossing_point,
            'nfcp': frequency_from_nfcp,
            'pk': frequency_from_peaks,
            'psd': frequency_from_psd,
        }

        for inx in range(0, len(prediction) - window_size, stride):
            prediction_window = prediction[inx:inx + window_size]
            ground_truth_window = ground_truth[inx:inx + window_size]

            for key, metric in metrics.items():
                self.prediction_results[key] = np.append(
                    self.prediction_results[key],
                    metric(prediction_window, sample_rate)
                )

                self.ground_truth_results[key] = np.append(
                    self.ground_truth_results[key],
                    metric(ground_truth_window, sample_rate)
                )

    def get_results(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Get the analysis results.
        :return: A dictionary containing the analysis results.
        """
        return self.prediction_results, self.ground_truth_results

    def compute_metrics(self) -> list[dict[str, float]]:
        """
        Compute the metrics for the analysis.
        :return: A dictionary containing the computed metrics.
        """
        metrics = []

        for key in self.prediction_results.keys():
            metrics.extend([{
                'metric': 'MSE',
                'method': key,
                'value': np.mean((self.prediction_results[key] - self.ground_truth_results[key]) ** 2)
            }, {
                'metric': 'MAE',
                'method': key,
                'value': np.mean(np.abs(self.prediction_results[key] - self.ground_truth_results[key]))
            }, {
                'metric': 'RMSE',
                'method': key,
                'value': np.sqrt(np.mean((self.prediction_results[key] - self.ground_truth_results[key]) ** 2))
            }, {
                'metric': 'MAPE',
                'method': key,
                'value': np.mean(np.abs(
                    (self.prediction_results[key] - self.ground_truth_results[key]) / self.ground_truth_results[
                        key])) * 100
            }, {
                'metric': 'PCC',
                'method': key,
                'value': pearson_correlation(self.prediction_results[key], self.ground_truth_results[key])
            }, {
                'metric': 'SCC',
                'method': key,
                'value': spearman_correlation(self.prediction_results[key], self.ground_truth_results[key])
            }])

        return metrics
