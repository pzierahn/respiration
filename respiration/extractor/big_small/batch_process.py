import torch.nn as nn

from tqdm import tqdm
from .frame_processing import *


def batch_process(
        model: nn.Module,
        frames: np.ndarray,
        device: torch.device,
        size: int = 360,
        show_progress: bool = False
) -> np.ndarray:
    waveform = None

    for inx in tqdm(range(0, len(frames), size), disable=(not show_progress)):
        end = min(inx + size, len(frames))
        big, small = preprocess_frames(frames[inx:end])
        big, small = convert_to_input(big, small, device=device)

        with torch.no_grad():
            _, _, resp_out = model((big, small))

            # Convert the signal to numpy
            resp_out = resp_out.cpu().numpy().squeeze()

            if waveform is None:
                waveform = resp_out
            else:
                waveform = np.concatenate((waveform, resp_out))

    return waveform.squeeze()
