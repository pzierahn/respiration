import os
import torch
from typing import Optional

from .big_small import BigSmall


def load_model(
        model_checkpoint: Optional[str] = None,
        device: torch.device = torch.device('cpu')
) -> tuple[torch.nn.Module, dict]:
    if model_checkpoint is None:
        model_checkpoint = os.path.join('..', 'data', 'rPPG-Toolbox', 'BP4D_BigSmall_Multitask_Fold3.pth')

    # Wrap modul in nn.DataParallel
    model = BigSmall()
    # Fix model loading: Some key have an extra 'module.' prefix
    model = torch.nn.DataParallel(model)
    model.to(device)

    key_matching = model.load_state_dict(torch.load(model_checkpoint, map_location=device))

    return model, key_matching
