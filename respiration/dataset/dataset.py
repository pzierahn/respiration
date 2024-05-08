import os
import re
import numpy as np
import respiration.utils as utils

from scipy.signal import resample
from typing import List, Optional


class VitalCamSet:
    """
    Class to handle the VitalCamSet dataset
    """

    # Path to the VitalCamSet dataset
    data_path: str

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            # Use the default path if not provided
            self.data_path = os.path.join('..', '..', 'data', 'VitalCamSet')
        else:
            self.data_path = data_path

    def get_subjects(self) -> List[str]:
        """
        Get the list of subjects in the dataset
        :return: list of subjects
        """

        files = os.listdir(self.data_path)

        # Only keep the folders with the format 'Proband[0-9]{2}'
        subjects = [file for file in files if re.match(r'Proband[0-9]{2}', file)]
        subjects = sorted(subjects)

        return subjects

    @staticmethod
    def get_settings() -> List[str]:
        return [
            '101_natural_lighting',
            '102_artificial_lighting',
            '103_abrupt_changing_lighting',
            '104_dim_lighting_auto_exposure',
            '106_green_lighting',
            '107_infrared_lighting',
            '201_shouldercheck',
            '202_scale_movement',
            '203_translation_movement',
            '204_writing'
        ]

    def get_scenarios(self, settings: Optional[list[str]] = None) -> List[tuple[str, str]]:
        if settings is None:
            settings = VitalCamSet.get_settings()

        scenarios = []

        for subject in self.get_subjects():
            for setting in settings:
                scenarios.append((subject, setting))

        return scenarios

    def get_video_path(self, subject: str, setting: str) -> str:
        """
        Get the path to the video file for a given subject and scenario
        :param subject: subject name
        :param setting: scenario name
        :return: path to the video file
        """

        subject_path = os.path.join(self.data_path, subject)
        scenario_path = os.path.join(subject_path, setting)
        video_path = os.path.join(scenario_path, 'Logitech HD Pro Webcam C920.avi')

        return video_path

    def get_first_frame(self, subject: str, setting: str) -> np.ndarray:
        """
        Get the first frame of a given subject and scenario
        :param subject: subject name
        :param setting: scenario name
        :return: numpy array of the first frame
        """

        video_path = self.get_video_path(subject, setting)
        frames, _ = utils.read_video_rgb(video_path, 1, 0)
        return frames[0]

    def get_video_gray(self, subject: str,
                       setting: str,
                       num_frames: Optional[int] = None,
                       start_position: int = 0,
                       progress: bool = True) -> tuple[np.ndarray, utils.VideoParams]:
        """
        Get the frames of a given subject and scenario in grayscale
        :param subject: subject name
        :param setting: scenario name
        :param num_frames: number of frames to read
        :param start_position: starting frame
        :param progress: whether to show progress bar
        :return: numpy array of frames and video parameters
        """

        video_path = self.get_video_path(subject, setting)
        return utils.read_video_gray(
            video_path,
            num_frames=num_frames,
            start_position=start_position,
            show_progress=progress)

    def get_video_bgr(self,
                      subject: str,
                      setting: str,
                      num_frames: Optional[int] = None,
                      start_position: int = 0,
                      show_progress: bool = False) -> tuple[np.ndarray, utils.VideoParams]:
        """
        Get the frames of a given subject and scenario in BGR
        :param subject: subject name
        :param setting: setting name
        :param num_frames: number of frames to read
        :param start_position: starting frame
        :param show_progress: whether to show progress bar
        :return: numpy array of frames and video parameters
        """

        video_path = self.get_video_path(subject, setting)
        return utils.read_video_bgr(video_path, num_frames, start_position, show_progress)

    def get_video_rgb(self, subject: str,
                      setting: str,
                      num_frames: Optional[int] = None,
                      start_position: int = 0,
                      show_progress: bool = False) -> tuple[np.ndarray, utils.VideoParams]:
        """
        Get the frames of a given subject and scenario in RGB
        :param subject: subject name
        :param setting: scenario name
        :param num_frames: number of frames to read
        :param start_position: starting frame
        :param show_progress: whether to show progress bar
        :return: numpy array of frames and video parameters
        """

        video_path = self.get_video_path(subject, setting)
        frames, meta = utils.read_video_bgr(video_path, num_frames, start_position, show_progress)
        return utils.bgr_to_rgb(frames), meta

    def get_unisens_entry(self, subject: str, setting: str, entry: utils.VitalSigns) -> tuple[np.ndarray, int]:
        """
        Read an entry from an unisens dataset
        :param subject: subject
        :param setting: setting
        :param entry: vital signal
        :return: numpy array of the signal and the sampling rate
        """

        subject_path = os.path.join(
            self.data_path,
            subject,
            setting,
            'synced_Logitech HD Pro Webcam C920')

        return utils.read_unisens_entry(subject_path, entry)

    def get_vital_sign(self, subject: str, setting: str, vital_sign: utils.VitalSigns) -> np.ndarray:
        """
        Get the ground truth signal for a given subject and scenario. The signal is resampled to match the video frame
        rate.
        """

        video_path = self.get_video_path(subject, setting)
        params = utils.get_video_params(video_path)
        signal, fs = self.get_unisens_entry(subject, setting, vital_sign)
        return resample(signal, params.num_frames)

    def get_breathing_signal(self, subject: str, setting: str) -> np.ndarray:
        """Get the ground truth respiratory signal for a given subject and scenario."""
        return self.get_vital_sign(subject, setting, utils.VitalSigns.thorax_abdomen)

    def contains(self, subject: str, setting: str) -> bool:
        """Check if a given subject and scenario exists in the dataset"""

        subject_path = os.path.join(
            self.data_path,
            subject,
            setting)

        return os.path.exists(subject_path)
