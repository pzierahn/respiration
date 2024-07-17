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
    """
    Class to perform an analysis of the respiration signals. The analysis includes the following steps:
    - Preprocess the signals (detrend, normalize, filter)
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
    detrend: bool
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
        Preprocess the prediction and ground truth signals. The preprocessing steps include detrending, filtering,
        and normalization.
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
                'SCC': np.array([]),
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

            self.distances[model]['SCC'] = np.append(
                self.distances[model]['SCC'],
                spearman_correlation(prediction_window, ground_truth_window)[0]
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

    def get_distances(self) -> list[dict]:
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

        return rows

    def distances_df(self) -> pd.DataFrame:
        """
        Generate a table with the computed distances.
        :return: A list of dictionaries containing the computed distances.
        """
        distances = self.get_distances()
        data = {}

        for entry in distances:
            model = entry['model']
            distance = entry['distance']

            if model not in data:
                data[model] = {}

            data[model][distance] = entry

        rows = []
        for model in data:
            row = {
                'model': model,
            }

            for distance in data[model]:
                row[distance] = data[model][distance]['mean']
                row[f'{distance}_sdt'] = data[model][distance]['std']

            rows.append(row)

        return pd.DataFrame(rows)

    def rank_models(self) -> pd.DataFrame:
        """
        Rank the models based on the computed metrics.
        :return: A DataFrame containing the ranking of the models.
        """
        metrics = self.get_metrics()
        rows = []

        for model in metrics:
            for method in metrics[model]:
                for metric in metrics[model][method]:
                    value = metrics[model][method][metric]

                    if metric == "PCC-p-value":
                        # Skip the p-values, because they should not be used for ranking
                        continue

                    if metric == "PCC":
                        # Bigger correlation values are better. We want to rank them in descending order.
                        # Therefore, we need to invert the value.
                        value = 1 - abs(value)

                    row = {
                        'model': model,
                        'method': method,
                        'metric': metric,
                        'value': abs(value),
                    }
                    rows.append(row)

        metrics_df = pd.DataFrame(rows)
        metrics_df['rank'] = metrics_df.groupby(['metric', 'method'])['value'].rank(ascending=True)

        return metrics_df
