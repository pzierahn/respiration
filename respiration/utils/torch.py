import torch


def get_torch_device() -> torch.device:
    """
    Get the torch device to run the model on
    :return:
    """

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Use the MPS (Multi-Process Service) to run the model
        # This is only available on macOS
        device = torch.device('mps')
    elif torch.cuda.is_available():
        # Use the GPU to run the model
        device = torch.device('cuda')
    else:
        # Use the CPU to run the model
        device = torch.device('cpu')

    return device
