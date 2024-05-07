import torch

from torchvision.models.optical_flow import (
    raft_large,
    raft_small,
    RAFT,
    Raft_Large_Weights,
    Raft_Small_Weights,
)


def load_model(model_name: str, device: torch.device) -> RAFT:
    """
    Load the optical flow model.
    :param model_name: The name of the model to load (either 'raft_large' or 'raft_small')
    :param device: The device to load the model on
    :return: The loaded model
    """

    if model_name == 'raft_large':
        model = raft_large(Raft_Large_Weights.C_T_V2)
    elif model_name == 'raft_small':
        model = raft_small(Raft_Small_Weights.C_T_V2)
    else:
        raise ValueError(f'Invalid model name: {model_name}')

    model = model.to(device)
    return model.eval()
