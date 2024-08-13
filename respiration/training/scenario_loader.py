import math
import torch

from respiration.utils.video import (
    get_frame_count,
    read_video_rgb,
)

from respiration.dataset import VitalCamSet


class ScenarioLoader:
    """
    A data loader for the VitalCamSet dataset. This class loads the video frames and the ground truth signal for a
    specific scenario. The video frames are loaded in chunks of a specific size. The ground truth signal is down-sampled
    to match the video frames' dimensions and normalized between -0.5 and 0.5.

    Version: 1.0
    """
    subject: str
    setting: str
    frames_per_segment: int

    def __init__(self,
                 subject: str,
                 setting: str,
                 frames_per_segment: int,
                 device: torch.device = torch.device("cpu")):
        self.subject = subject
        self.setting = setting
        self.device = device
        self.frames_per_segment = frames_per_segment

        self.dataset = VitalCamSet()
        self.video_path = self.dataset.get_video_path(subject, setting)
        self.total_frames = get_frame_count(self.video_path)

    def __len__(self) -> int:
        return math.ceil(self.total_frames / self.frames_per_segment)

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

        start = index * self.frames_per_segment
        end = start + self.frames_per_segment
        size = min(self.frames_per_segment, self.total_frames - start)

        # Load the video frames
        frames, meta = read_video_rgb(self.video_path, size, start)
        frames = torch.tensor(frames, dtype=torch.float32, device=self.device)

        # Permute the dimensions to match the expected input format (B, C, H, W)
        frames = frames.permute(0, 3, 1, 2)

        # Get the ground truth signal for the scenario
        gt_waveform = self.dataset.get_breathing_signal(self.subject, self.setting)
        gt_waveform = torch.tensor(gt_waveform.copy(), dtype=torch.float32, device=self.device)

        #
        # Normalize the signals: This normalizes the signal between -0.5 and 0.5. The values are based on the
        # overall maximum and minimum values in the dataset.
        #

        # The absolute maximum and minimum values in the dataset.
        gt_overall_max, gt_overall_min = 6680.352219172085, -6572.075174276201
        gt_waveform = (gt_waveform - gt_waveform.mean()) / (gt_overall_max - gt_overall_min)
        gt_waveform = gt_waveform[start:end]

        return frames, gt_waveform


class VitalCamLoader:
    """
    A data loader for the VitalCamSet dataset. This class loads the video frames and the ground truth signal for a list
    of subjects and settings. The video frames split into parts and loaded in chunks.

    Version: 1.0
    """

    # The list of subjects and settings to load
    scenarios: list[tuple[str, str]]

    # The number of frames per video
    total_frames = 3600

    # The number of parts to split the video into
    parts: int

    # The device to use for the tensors
    device: torch.device

    def __init__(self,
                 scenarios: list[tuple[str, str]],
                 total_frames: int = 3600,
                 parts: int = 10,
                 device: torch.device = torch.device("cpu")):
        self.scenarios = scenarios
        self.total_frames = total_frames
        self.parts = parts
        self.device = device

    def __len__(self) -> int:
        """
        Return the number of items in the data loader. This is the number of scenarios times the number of parts.
        """
        return len(self.scenarios) * self.parts

    def __iter__(self):
        """
        Initialize the data loader
        """
        self.current_index = 0
        return self

    def __next__(self):
        """
        Return the next item in the data loader
        """
        if self.current_index >= self.__len__():
            raise StopIteration
        else:
            item = self.__getitem__(self.current_index)
            self.current_index += 1
            return item

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        """
        Return the frames and the ground truth signal for the given index
        """
        if index >= self.__len__():
            raise IndexError("Index out of range")

        subject_index = index // self.parts
        setting_index = index % self.parts

        subject, setting = self.scenarios[subject_index]
        scenario_loader = ScenarioLoader(subject, setting, self.total_frames // self.parts, self.device)
        return scenario_loader[setting_index]
