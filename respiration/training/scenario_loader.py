import math
import torch
from torchvision import transforms

import respiration.utils as utils
import respiration.dataset as ds


class ScenarioLoader:
    """
    A data loader for the VitalCamSet dataset. This class loads the video frames and the ground truth signal for a
    specific scenario. The video frames are loaded in chunks of a specific size. The ground truth signal is down-sampled
    to match the video frames' dimensions.
    """
    subject: str
    setting: str
    chunk_size: int
    dataset: ds.VitalCamSet
    video_path: str
    total_frames: int
    image_size: int

    def __init__(self,
                 dataset: ds.VitalCamSet,
                 subject: str,
                 setting: str,
                 chunk_size: int,
                 image_size: int,
                 device: torch.device):
        self.subject = subject
        self.setting = setting
        self.chunk_size = chunk_size + 1
        self.dataset = dataset
        self.image_size = image_size

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.video_path = dataset.get_video_path(subject, setting)
        self.total_frames = utils.get_frame_count(self.video_path)

    def __len__(self) -> int:
        return math.ceil(self.total_frames / self.chunk_size)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.__len__():
            raise StopIteration
        else:
            item = self.__getitem__(self.current_index)
            self.current_index += 1
            return item

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        """
        Return the frames and the ground truth signal for the given index
        :param index: The index of the chunk
        :return: The frames and the ground truth signal
        """

        if index >= self.__len__():
            raise IndexError("Index out of range")

        start = index * self.chunk_size
        end = start + self.chunk_size
        size = min(self.chunk_size, self.total_frames - start)

        # Load the video frames
        frames, _ = utils.read_video_rgb(self.video_path, size, start)
        preprocess = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        frames = torch.stack([preprocess(frame) for frame in frames], dim=0)
        frames = frames.to(self.device)

        # Get the ground truth signal for the scenario
        gt_waveform = self.dataset.get_breathing_signal(self.subject, self.setting)
        gt_waveform = torch.tensor(gt_waveform, dtype=torch.float32, device=self.device)
        gt_waveform = torch.nn.functional.normalize(gt_waveform, dim=0)
        gt_waveform = gt_waveform[start:end]

        return frames, gt_waveform
