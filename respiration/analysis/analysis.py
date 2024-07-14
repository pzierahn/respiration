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
            self.prediction_metrics[model] = {
                key: np.array([]) for key in metrics.keys()
            }
            self.ground_truth_metrics[model] = {
                key: np.array([]) for key in metrics.keys()
            }

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

    def compute_metrics(self) -> list[dict[str, float]]:
        """
        Compute the metrics for the analysis.
        :return: A dictionary containing the computed metrics.
        """
        metrics = []

        for model in self.prediction_metrics.keys():
            for method in self.prediction_metrics[model].keys():
                metrics.extend([{
                    'model': model,
                    'method': method,
                    'metric': 'MSE',
                    'value': np.mean(
                        (self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method]) ** 2)
                }, {
                    'model': model,
                    'method': method,
                    'metric': 'MAE',
                    'value': np.mean(
                        np.abs(self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method]))
                }, {
                    'model': model,
                    'method': method,
                    'metric': 'RMSE',
                    'value': np.sqrt(
                        np.mean(
                            (self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method]) ** 2))
                }, {
                    'model': model,
                    'method': method,
                    'metric': 'MAPE',
                    'value': np.mean(np.abs(
                        (self.prediction_metrics[model][method] - self.ground_truth_metrics[model][method]) /
                        self.ground_truth_metrics[
                            model][method])) * 100
                }, {
                    'model': model,
                    'method': method,
                    'metric': 'PCC',
                    'value': pearson_correlation(
                        self.prediction_metrics[model][method],
                        self.ground_truth_metrics[model][method])
                }, {
                    'model': model,
                    'method': method,
                    'metric': 'SCC',
                    'value': spearman_correlation(
                        self.prediction_metrics[model][method],
                        self.ground_truth_metrics[model][method])
                }])

        return metrics

    def metric_table(self) -> list[dict[str, float]]:
        """
        Generate a table with the computed metrics.
        :return: A list of dictionaries containing the computed metrics.
        """
        metrics = self.compute_metrics()
        table = {}

        for metric in metrics:
            model = metric['model']
            method = metric['method']

            if model not in table:
                table[model] = {}
            if method not in table[model]:
                table[model][method] = {
                    'model': model,
                    'method': method,
                }

            table[model][method][metric['metric']] = metric['value']

        entries = []
        for value in table.values():
            entries.extend([entry for entry in value.values()])

        return entries

    def rank_models(self, show_metrics: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Rank the models based on the computed metrics.
        :param show_metrics: The metrics to show in the ranking. If None, all metrics are shown.
        :return: A DataFrame containing the ranking of the models.
        """
        metrics = self.compute_metrics()
        analysis_results = pd.DataFrame(metrics)

        if show_metrics is None:
            # Show all metrics if not specified
            show_metrics = analysis_results['metric'].unique()

        # Keep only the metrics that are specified
        analysis_results = analysis_results[analysis_results['metric'].isin(show_metrics)]

        # A higher correlation value is better. Hence, we need to invert the correlation values
        # to rank the models.
        corr_loc = (analysis_results['metric'] == 'SCC') | (analysis_results['metric'] == 'PCC')
        analysis_results.loc[corr_loc, 'value'] = analysis_results.loc[corr_loc, 'value'].abs()
        analysis_results.loc[corr_loc, 'value'] = 1 - analysis_results.loc[corr_loc, 'value']

        # Rank the models based on the mean of the metrics
        # Add new rank column
        analysis_results['rank'] = 0

        metrics = analysis_results['metric'].unique()
        methods = analysis_results['method'].unique()

        for metric in metrics:
            for method in methods:
                loc = ((analysis_results['method'] == method) &
                       (analysis_results['metric'] == metric))

                ranks = analysis_results[loc]['value'].rank().astype(int)
                analysis_results.loc[loc, 'rank'] = ranks

        return analysis_results

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
