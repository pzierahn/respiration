import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_sampling(mean: torch.Tensor, std: torch.Tensor, label_k: torch.Tensor):
    """
    Compute the normal distribution for a given mean and standard deviation.
    :param mean: Mean of the normal distribution
    :param std: Standard deviation of the normal distribution
    :param label_k: Label for the normal distribution
    :return: Normal distribution
    """
    return torch.exp(-((label_k - mean) ** 2) / (2 * std ** 2)) / (torch.sqrt(torch.tensor(2 * torch.pi)) * std)


def filtered_periodogram(
        time_series: torch.Tensor,
        sampling_rate: int,
        min_freq: float = 0,
        max_freq: float = float('inf')) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the power spectral density (PSD) of a signal within a given frequency range.
    :param time_series: Respiratory signal
    :param sampling_rate: Sampling rate
    :param min_freq: minimum frequency
    :param max_freq: maximum frequency
    :return: Frequencies and FFT result
    """

    # Compute the power spectral density (PSD) using periodogram
    psd = (torch.fft.fft(time_series).abs() ** 2) / time_series.shape[0]

    psd = psd[:len(psd) // 2]
    freq = torch.fft.fftfreq(time_series.shape[0], 1 / sampling_rate)[:len(psd)]

    # Find the indices corresponding to the frequency range
    idx = (freq >= min_freq) & (freq <= max_freq)

    # Extract the frequencies and PSDs within the specified range
    freq_range = freq[idx]
    psd_range = psd[idx]

    # Make the psd sum to 1
    psd_range = psd_range / psd_range.sum()

    return freq_range, psd_range


def euclidean_distance(pred_psd: torch.Tensor, gt_psd: torch.Tensor):
    """
    Compute the Euclidean distance between the predicted and ground truth power spectral densities.
    :param pred_psd: Predicted power spectral density
    :param gt_psd: Ground truth power spectral density
    :return: Euclidean distance
    """
    return torch.dist(pred_psd.softmax(dim=0), gt_psd.softmax(dim=0))


def cosine_distance(pred_psd: torch.Tensor, gt_psd: torch.Tensor):
    """
    Compute the cosine distance between the predicted and ground truth power spectral densities.
    :param pred_psd: Predicted power spectral density
    :param gt_psd: Ground truth power spectral density
    :return: Cosine distance
    """
    return 1 - F.cosine_similarity(pred_psd, gt_psd, dim=0)


def frequency_loss(pred_psd: torch.Tensor, gt_psd: torch.Tensor):
    """
    Compute the cross-entropy loss between the predicted and ground truth power spectral densities.
    :param pred_psd: Predicted power spectral density
    :param gt_psd: Ground truth power spectral density
    :return: Cross-entropy loss
    """
    return F.cross_entropy(pred_psd, torch.argmax(gt_psd))


def pearson_correlation(prediction: torch.Tensor, ground_truth: torch.Tensor):
    """
    Compute Pearson correlation coefficient
    :param prediction: Predicted respiratory signal
    :param ground_truth: Ground truth respiratory signal
    :return: Pearson correlation coefficient
    """
    x_mean = torch.mean(prediction)
    y_mean = torch.mean(ground_truth)

    num = torch.sum((prediction - x_mean) * (ground_truth - y_mean))
    den = torch.sqrt(torch.sum((prediction - x_mean) ** 2) * torch.sum((ground_truth - y_mean) ** 2))

    correlation = num / den

    # Bigger correlation means smaller loss, so we negate it
    return 1 - correlation


def norm_kl_loss(pred_psd: torch.Tensor, gt_psd: torch.Tensor, std=torch.tensor(3.0)) -> torch.Tensor:
    """
    Compute the Kullback-Leibler divergence between two normal distributions.
    :param pred_psd: Predicted power spectral density
    :param gt_psd: Ground truth power spectral density
    :param std: Standard deviation of the normal distribution
    :return: Kullback-Leibler divergence between
    """

    pred_mean = torch.argmax(pred_psd)
    pred_label = torch.arange(pred_psd.shape[0], device=pred_psd.device)
    pred_norm = normal_sampling(pred_mean, std, pred_label)

    gt_mean = torch.argmax(gt_psd)
    gt_label = torch.arange(gt_psd.shape[0], device=gt_psd.device)
    gt_norm = normal_sampling(gt_mean, std, gt_label)

    criterion = torch.nn.KLDivLoss(reduction='none')
    return criterion(pred_norm.log(), gt_norm).sum()


class HybridLoss(nn.Module):
    """
    Hybrid loss function combining temporal loss (Pearson correlation), frequency loss and norm loss.
    """

    # Sampling rate of the signal
    sampling_rate: int

    # Frequency range for filtering the signal
    min_freq: float
    max_freq: float

    pearson_weight: float
    frequency_weight: float
    norm_weight: float
    mse_weight: float

    def __init__(
            self,
            sampling_rate: int = 30,
            min_freq: float = 0.08,
            max_freq: float = 0.6,
            pearson_weight: float = 0.2,
            frequency_weight: float = 1.0,
            norm_weight: float = 1.0,
            mse_weight: float = 1.0
    ):
        super(HybridLoss, self).__init__()

        self.sampling_rate = sampling_rate
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.pearson_weight = pearson_weight
        self.frequency_weight = frequency_weight
        self.norm_weight = norm_weight
        self.mse_weight = mse_weight

    def forward(self, prediction, ground_truth):
        """Compute the hybrid loss"""
        pearson = pearson_correlation(prediction, ground_truth)

        freq_pred, pred_psd = filtered_periodogram(prediction, self.sampling_rate, self.min_freq, self.max_freq)
        freq_gt, gt_psd = filtered_periodogram(ground_truth, self.sampling_rate, self.min_freq, self.max_freq)

        freq_loss = frequency_loss(pred_psd, gt_psd)
        norm_l = norm_kl_loss(pred_psd, gt_psd)

        mse = F.mse_loss(prediction, ground_truth)

        # Combine losses
        total_loss = (self.pearson_weight * pearson +
                      self.frequency_weight * freq_loss +
                      self.norm_weight * norm_l +
                      self.mse_weight * mse)

        return total_loss
