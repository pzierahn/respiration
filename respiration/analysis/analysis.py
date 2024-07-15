import numpy as np
import pandas as pd

from typing import Optional

from .cross_point import (
    frequency_from_crossing_point,
    frequency_from_nfcp,
)
from .distance import (
    pearson_correlation,
    spearman_correlation,
    dtw_distance,
)
from .peak_counting import frequency_from_peaks
from .psd import frequency_from_psd
from .preprocessing import *


class Analysis:
    sample_rate: int

    lowpass: Optional[float]
    highpass: Optional[float]

    detrend: bool
    normalize: bool
    filter_signal: bool

    window_size: int
    stride: int

    # Analysis results: Model Name --> Method --> Results
    prediction_metrics: dict[str, dict[str, np.ndarray]]
    ground_truth_metrics: dict[str, dict[str, np.ndarray]]

    # Raw data: Model Name --> Signal
    predictions: dict[str, np.ndarray]
    ground_truths: dict[str, np.ndarray]

    # Distances between signals
    distances: dict[str, dict[str, np.ndarray]]

    def __init__(
            self,
            sample_rate: int,
            lowpass: Optional[float] = 0.08,
            highpass: Optional[float] = 0.6,
            detrend: bool = False,
            normalize: bool = True,
            filter_signal: bool = True,
            window_size: int = 30,
            stride: int = 1
    ):
        self.sample_rate = sample_rate

        self.lowpass = lowpass
        self.highpass = highpass

        self.detrend = detrend
        self.normalize = normalize
        self.filter_signal = filter_signal

        self.window_size = window_size
        self.stride = stride

        self.prediction_metrics = {}
        self.ground_truth_metrics = {}

        self.predictions = {}
        self.ground_truths = {}

        self.distances = {}

    def __preprocess(
            self,
            prediction: np.ndarray,
            ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the prediction and ground truth signals.
        :param prediction: The predicted signal.
        :param ground_truth: The ground truth signal.
        :return: The preprocessed prediction and ground truth signals.
        """

        assert prediction.shape == ground_truth.shape, \
            (f'Prediction and ground truth signals must have the same shape. Got prediction shape: {prediction.shape}, '
             f'ground truth shape: {ground_truth.shape}')

        if self.detrend:
            prediction = detrend_tarvainen(prediction)
            ground_truth = detrend_tarvainen(ground_truth)

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
            'pk': frequency_from_peaks,
            'psd': frequency_from_psd,
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

            self.distances[model] = {
                'dtw': np.array([]),
                'scc': np.array([]),
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

            self.distances[model]['dtw'] = np.append(
                self.distances[model]['dtw'],
                dtw_distance(prediction_window, ground_truth_window)
            )
            self.distances[model]['scc'] = np.append(
                self.distances[model]['scc'],
                spearman_correlation(prediction_window, ground_truth_window)[0]
            )

    def correlation_signals(self):
        """
        Compute the correlations for the signals.
        """
        correlations = []

        for model in self.prediction_metrics.keys():
            pcc, p_pcc = pearson_correlation(
                self.predictions[model],
                self.ground_truths[model]
            )

            scc, p_scc = spearman_correlation(
                self.predictions[model],
                self.ground_truths[model]
            )

            correlations.extend([{
                'model': model,
                'correlation': 'PCC',
                'statistic': pcc,
                'p-value': pcc,
            }, {
                'model': model,
                'correlation': 'SCC',
                'statistic': scc,
                'p-value': p_scc,
            }])

        return correlations

    def compute_metrics(self) -> dict[str, dict[str, dict[str, float]]]:
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
                    self.predictions[model],
                    self.ground_truths[model]
                )

                metrics[model][method] = {
                    'MSE': np.mean(
                        (self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method]) ** 2
                    ),
                    'MAE': np.mean(
                        np.abs(self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method])
                    ),
                    'RMSE': np.sqrt(
                        np.mean(
                            (self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method]) ** 2
                        )
                    ),
                    'MAPE': np.mean(np.abs(
                        (self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method]) /
                        self.ground_truth_metrics[model][method]
                    )) * 100,
                    'PCC': pcc,
                    'PCC-p-value': p_pcc,
                }

        return metrics

    def metrics_df(self) -> pd.DataFrame:
        """
        Generate a table with the computed metrics.
        :return: A list of dictionaries containing the computed metrics.
        """
        metrics = self.compute_metrics()
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

    def rank_models(self) -> pd.DataFrame:
        """
        Rank the models based on the computed metrics.
        :param show_metrics: The metrics to show in the ranking. If None, all metrics are shown.
        :return: A DataFrame containing the ranking of the models.
        """
        metrics = self.compute_metrics()
        rows = []

        for model in metrics:
            for method in metrics[model]:
                for metric in metrics[model][method]:
                    value = metrics[model][method][metric]

                    if metric is "PCC-p-value":
                        # Skip the p-values, because they should not be used for ranking
                        continue

                    row = {
                        'model': model,
                        'method': method,
                        'metric': metric,
                        'value': abs(value),  # Correlation values need to be positive
                    }
                    rows.append(row)

        metrics_df = pd.DataFrame(rows)
        metrics_df['rank'] = metrics_df.groupby(['metric', 'method'])['value'].rank(ascending=True)

        return metrics_df

    def distances_df(self) -> pd.DataFrame:
        """
        Generate a table with the computed distances.
        :return: A list of dictionaries containing the computed distances.
        """
        rows = []

        for model in self.distances:
            for distance in self.distances[model]:
                row = {
                    'model': model,
                    'distance': distance,
                    'mean': self.distances[model][distance].mean(),
                    'std': self.distances[model][distance].std(),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def get_mean_model_ranks(self, show_metrics: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Get the scores of the models based on the computed metrics.
        :param show_metrics: The metrics to show in the ranking. If None, all metrics are shown.
        :return: A DataFrame containing the scores of the model.
        """
        analysis_results = self.rank_models(show_metrics)

        # Show the mean rank for each model
        mean_rank = analysis_results.groupby('model')['rank'].mean().sort_values()
        mean_rank = mean_rank.rename('mean_rank')

        # Add the standard deviation
        std_rank = analysis_results.groupby('model')['rank'].std().sort_values()
        std_rank = std_rank.rename('std_rank')

        mean_rank = pd.concat([mean_rank, std_rank], axis=1)
        return mean_rank
