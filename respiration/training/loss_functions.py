import torch
from torch import nn
from abc import abstractmethod
from tslearn.metrics import SoftDTWLossPyTorch


class _LossFunction(nn.Module):
    def __init__(self):
        super(_LossFunction, self).__init__()

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass


class PearsonLoss(_LossFunction):
    def name(self):
        return 'pearson'

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Pearson loss between the true and predicted signals using PyTorch.
        :param inputs: The predicted signal
        :param targets: The true signal
        :return: The Pearson loss
        """

        # Ensure the signals have the same length
        assert targets.size(0) == inputs.size(0), "Signals must have the same length"

        vx = inputs - torch.mean(inputs)
        vy = targets - torch.mean(targets)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - cost


class MeanSquaredError(_LossFunction):
    def name(self):
        return 'mse'

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Mean Squared Error loss between the true and predicted signals using PyTorch.
        :param inputs: The predicted signal
        :param targets: The true signal
        :return: The Mean Squared Error loss
        """
        criterion = nn.MSELoss()
        return criterion(inputs, targets)


class SoftDWT(_LossFunction):
    def name(self):
        return 'soft_dwt'

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Cross Entropy loss between the true and predicted signals using PyTorch.
        :param inputs: The predicted signal
        :param targets: The true signal
        :return: The Mean Squared Error loss
        """

        # Transform the signals to the shape batch_size x 1 x sequence_length
        inputs = inputs.reshape(1, inputs.shape[0], 1)
        targets = targets.reshape(1, targets.shape[0], 1)

        criterion = SoftDTWLossPyTorch(gamma=0.1)
        return criterion(inputs, targets).abs().mean()
