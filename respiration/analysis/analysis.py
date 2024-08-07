import pandas as pd

from typing import Optional

from .cross_point import (
    frequency_from_crossing_point,
    frequency_from_nfcp,
)
from .distance import (
    pearson_correlation,
)

from .peak_counting import frequency_from_peaks
from .psd import frequency_from_psd
from .preprocessing import *


class Analysis:
    """
    The Analysis class is used to compare the performance of different models in predicting respiration signals. The
    analysis includes the following steps:
    - Preprocess the signals (normalize, filter)
    - Compute the metrics for the signals
    - Compute the distances between the signals
    - Rank the models based on the computed metrics
    """
    # Signal parameters
    sample_rate: int

    # Bandpass filter parameters
    lowpass: Optional[float]
    highpass: Optional[float]

    # Preprocessing parameters
    normalize: bool
    filter_signal: bool

    # Parameters for the sliding window analysis
    window_size: int
    stride: int

    # Analysis results: Model Name --> Method --> Results
    prediction_metrics: dict[str, dict[str, np.ndarray]]
    ground_truth_metrics: dict[str, dict[str, np.ndarray]]

    # Raw data: Model Name --> Signal
    predictions: dict[str, np.ndarray]
    ground_truths: dict[str, np.ndarray]

    def __init__(
            self,
            sample_rate: int = 30,
            lowpass: Optional[float] = 0.08,
            highpass: Optional[float] = 0.6,
            normalize: bool = True,
            filter_signal: bool = True,
            window_size: int = 30,
            stride: int = 1
    ):
        """
        Initialize the Analysis object.
        :param sample_rate: The sample rate of the signals.
        :param lowpass: The lowpass frequency for the bandpass filter.
        :param highpass: The highpass frequency for the bandpass filter.
        :param normalize: If the signals should be normalized.
        :param filter_signal: If the signals should be filtered.
        :param window_size: The size of the sliding window in seconds.
        :param stride: The stride of the sliding window in seconds.
        """
        self.sample_rate = sample_rate

        self.lowpass = lowpass
        self.highpass = highpass

        self.normalize = normalize
        self.filter_signal = filter_signal

        self.window_size = window_size
        self.stride = stride

        self.prediction_metrics = {}
        self.ground_truth_metrics = {}

        self.predictions = {}
        self.ground_truths = {}

    def __preprocess(
            self,
            prediction: np.ndarray,
            ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the prediction and ground truth signals. The preprocessing steps include detrending, filtering,
        and normalization.
        :param prediction: The predicted signal.
        :param ground_truth: The ground truth signal.
        :return: The preprocessed prediction and ground truth signals.
        """

        assert prediction.shape == ground_truth.shape, \
            (f'Prediction and ground truth signals must have the same shape. Got prediction shape: {prediction.shape}, '
             f'ground truth shape: {ground_truth.shape}')

        if self.filter_signal:
            prediction = butterworth_filter(
                prediction,
                self.sample_rate,
                self.lowpass,
                self.highpass)
            ground_truth = butterworth_filter(
                ground_truth,
                self.sample_rate,
                self.lowpass,
                self.highpass)

        if self.normalize:
            prediction = normalize_signal(prediction)
            ground_truth = normalize_signal(ground_truth)

        return prediction, ground_truth

    def frequency_from_peaks(self, data: np.ndarray, sample_rate: int) -> float:
        """
        Compute the respiration frequency from the peaks of the signal.
        :param data: The signal.
        :param sample_rate: The sample rate of the signal.
        :return: The respiration frequency.
        """
        return frequency_from_peaks(data, sample_rate, min_frequency=self.lowpass)

    def frequency_from_psd(self, data: np.ndarray, sample_rate: int) -> float:
        """
        Compute the respiration frequency from the peaks of the signal.
        :param data: The signal.
        :param sample_rate: The sample rate of the signal.
        :return: The respiration frequency.
        """
        return frequency_from_psd(data, sample_rate, min_freq=self.lowpass, max_freq=self.highpass)

    def add_data(self, model: str, prediction: np.ndarray, ground_truth: np.ndarray):
        """
        Add data to the analysis.
        :param model: The model used to generate the prediction.
        :param prediction: The predicted signal.
        :param ground_truth: The ground truth signal.
        """
        prediction, ground_truth = self.__preprocess(prediction, ground_truth)

        # Calculate the window size and stride in samples
        window_size = self.window_size * self.sample_rate
        stride = self.stride * self.sample_rate

        metrics = {
            'cp': frequency_from_crossing_point,
            'nfcp': frequency_from_nfcp,
            'pk': self.frequency_from_peaks,
            'psd': self.frequency_from_psd,
        }

        if model not in self.prediction_metrics:
            self.predictions[model] = np.array([])
            self.ground_truths[model] = np.array([])

            self.prediction_metrics[model] = {
                key: np.array([]) for key in metrics.keys()
            }
            self.ground_truth_metrics[model] = {
                key: np.array([]) for key in metrics.keys()
            }

        self.predictions[model] = np.append(self.predictions[model], prediction)
        self.ground_truths[model] = np.append(self.ground_truths[model], ground_truth)

        for inx in range(0, len(prediction) - window_size, stride):
            prediction_window = prediction[inx:inx + window_size]
            ground_truth_window = ground_truth[inx:inx + window_size]

            for key, metric in metrics.items():
                self.prediction_metrics[model][key] = np.append(
                    self.prediction_metrics[model][key],
                    metric(prediction_window, self.sample_rate)
                )

                self.ground_truth_metrics[model][key] = np.append(
                    self.ground_truth_metrics[model][key],
                    metric(ground_truth_window, self.sample_rate)
                )

    def add_data_without_window(self, model: str, prediction: np.ndarray, ground_truth: np.ndarray):
        """
        Add data to the analysis.
        :param model: The model used to generate the prediction.
        :param prediction: The predicted signal.
        :param ground_truth: The ground truth signal.
        """
        prediction, ground_truth = self.__preprocess(prediction, ground_truth)

        metrics = {
            'cp': frequency_from_crossing_point,
            'nfcp': frequency_from_nfcp,
            'pk': self.frequency_from_peaks,
            'psd': self.frequency_from_psd,
        }

        if model not in self.prediction_metrics:
            self.predictions[model] = np.array([])
            self.ground_truths[model] = np.array([])

            self.prediction_metrics[model] = {
                key: np.array([]) for key in metrics.keys()
            }
            self.ground_truth_metrics[model] = {
                key: np.array([]) for key in metrics.keys()
            }

        self.predictions[model] = np.append(self.predictions[model], prediction)
        self.ground_truths[model] = np.append(self.ground_truths[model], ground_truth)

        for key, metric in metrics.items():
            self.prediction_metrics[model][key] = np.append(
                self.prediction_metrics[model][key],
                metric(prediction, self.sample_rate)
            )

            self.ground_truth_metrics[model][key] = np.append(
                self.ground_truth_metrics[model][key],
                metric(ground_truth, self.sample_rate)
            )

    def get_metrics(self) -> dict[str, dict[str, dict[str, float]]]:
        """
        Compute the metrics for the analysis.
        :return: A dictionary containing the computed metrics.
        """
        metrics = {}

        for model in self.prediction_metrics.keys():
            metrics[model] = {}
            for method in self.prediction_metrics[model].keys():
                metrics[model][method] = {}

        for model in self.prediction_metrics.keys():
            for method in self.prediction_metrics[model].keys():
                pcc, p_pcc = pearson_correlation(
                    self.prediction_metrics[model][method],
                    self.ground_truth_metrics[model][method],
                )

                metrics[model][method] = {
                    'MAE': np.mean(
                        np.abs(self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method])
                    ),
                    'RMSE': np.sqrt(np.mean(
                        (self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method]) ** 2
                    )),
                    'PCC': pcc,
                    'PCC-p-value': p_pcc,
                }

        return metrics

    def metrics_df(self) -> pd.DataFrame:
        """
        Generate a table with the computed metrics.
        :return: A list of dictionaries containing the computed metrics.
        """
        metrics = self.get_metrics()
        rows = []

        for model in metrics:
            for method in metrics[model]:
                row = {
                    'model': model,
                    'method': method,
                }

                for metric in metrics[model][method]:
                    row[metric] = metrics[model][method][metric]

                rows.append(row)

        return pd.DataFrame(rows)
