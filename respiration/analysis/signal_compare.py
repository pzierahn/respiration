from typing import Optional
from dtaidistance import dtw

from .psd import *
from .cross_point import *
from .peak_counting import *
from .distance import *

import respiration.preprocessing as preprocessing


class SignalComparator:
    """
    Analyze and compare prediction and ground truth signals in terms of beats per minute,
    error metrics, and signal distance measures.
    """

    prediction: np.ndarray
    ground_truth: np.ndarray
    sample_rate: int

    lowpass: Optional[float]
    highpass: Optional[float]

    round_decimals: int

    def __init__(
            self,
            prediction: np.ndarray,
            ground_truth: np.ndarray,
            sample_rate: int,
            lowpass: Optional[float] = 0.08,
            highpass: Optional[float] = 0.6,
            detrend_tarvainen: bool = True,
            normalize_signal: bool = True,
            filter_signal: bool = True,
            round_decimals: int = 0
    ):
        assert prediction.shape == ground_truth.shape, \
            (f'Prediction and ground truth signals must have the same shape. Got prediction shape: {prediction.shape}, '
             f'ground truth shape: {ground_truth.shape}')

        if detrend_tarvainen:
            prediction = preprocessing.detrend_tarvainen(prediction)
            ground_truth = preprocessing.detrend_tarvainen(ground_truth)

        if filter_signal:
            prediction = preprocessing.butterworth_filter(prediction, sample_rate, lowpass, highpass)
            ground_truth = preprocessing.butterworth_filter(ground_truth, sample_rate, lowpass, highpass)

        if normalize_signal:
            prediction = preprocessing.normalize_signal(prediction)
            ground_truth = preprocessing.normalize_signal(ground_truth)

        self.lowpass = lowpass
        self.highpass = highpass
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.sample_rate = sample_rate
        self.round_decimals = round_decimals

    def __frequency_to_bmp(self, gt_frequency: float, pred_frequency: float) -> tuple[float, float]:
        """
        Convert the frequency to beats per minute (bpm).
        """
        bmp_gt = round(gt_frequency * 60, self.round_decimals)
        bmp_pred = round(pred_frequency * 60, self.round_decimals)

        return bmp_gt, bmp_pred

    def psd(self) -> tuple[float, float]:
        """Get the bpm for the ground truth and the prediction using the power spectral density (psd) method."""

        gt_frequency = frequency_from_psd(
            self.ground_truth,
            self.sample_rate,
            self.lowpass,
            self.highpass,
        )
        pred_frequency = frequency_from_psd(
            self.prediction,
            self.sample_rate,
            self.lowpass,
            self.highpass,
        )

        return self.__frequency_to_bmp(gt_frequency, pred_frequency)

    def pc(self) -> tuple[float, float]:
        """Get the bpm for the ground truth and the prediction using the peak counting method."""
        gt_frequency = frequency_from_peaks(self.ground_truth, self.sample_rate)
        pred_frequency = frequency_from_peaks(self.prediction, self.sample_rate)

        return self.__frequency_to_bmp(gt_frequency, pred_frequency)

    def cp(self) -> tuple[float, float]:
        """Get the bpm for the ground truth and the prediction using the crossing point method."""
        gt_frequency = frequency_from_crossing_point(self.ground_truth, self.sample_rate)
        pred_frequency = frequency_from_crossing_point(self.prediction, self.sample_rate)

        return self.__frequency_to_bmp(gt_frequency, pred_frequency)

    def nfcp(self) -> tuple[float, float]:
        """
        Get the bpm for the ground truth and the prediction using the negative first crossing point (nfcp) method.
        """

        gt_frequency = frequency_from_nfcp(self.ground_truth, self.sample_rate)
        pred_frequency = frequency_from_nfcp(self.prediction, self.sample_rate)

        return self.__frequency_to_bmp(gt_frequency, pred_frequency)

    def all_results(self) -> dict[str, dict[str, float]]:
        """
        Compare the ground truth and the prediction signals using different methods.
        """
        pk_gt, pk_pred = self.pc()
        cp_gt, cp_pred = self.cp()
        nfcp_gt, nfcp_pred = self.nfcp()
        psd_gt, psd_pred = self.psd()

        return {
            'pk': {
                'ground_truth': pk_gt,
                'prediction': pk_pred,
            },
            'cp': {
                'ground_truth': cp_gt,
                'prediction': cp_pred,
            },
            'nfcp': {
                'ground_truth': nfcp_gt,
                'prediction': nfcp_pred,
            },
            'psd': {
                'ground_truth': psd_gt,
                'prediction': psd_pred,
            },
        }

    def errors(self) -> dict[str, float]:
        """
        Calculate the errors between the ground truth and the predictions for each method.
        :return: Dictionary with the errors for each method in beats per minute.
        """
        results = {}
        for metric, result in self.all_results().items():
            gt = result['ground_truth']
            pred = result['prediction']
            results[f'{metric}_error'] = abs(gt - pred)

        return results

    def distance_mse(self) -> float:
        """Calculate the mean absolute error between the two signals."""
        return distance_mse(self.ground_truth, self.prediction)

    def distance_dtw(self) -> float:
        """Calculate the mean absolute error between the two signals."""
        return dtw_distance(self.ground_truth, self.prediction)

    def distance_pearson(self) -> float:
        """Calculate the mean absolute error between the two signals."""
        return pearson_correlation(self.ground_truth, self.prediction)

    def signal_distances(self) -> dict[str, float]:
        """
        Calculate the distances between the ground truth and the prediction signals.
        """

        return {
            'mse': self.distance_mse(),
            'dtw': self.distance_dtw(),
            'pearson': self.distance_pearson(),
        }
