import os
import re
import numpy as np
import respiration.utils as utils

from typing import List, Optional


def from_path(data_path: str) -> 'Dataset':
    """
    Create a new Dataset object from a given path
    :param data_path: to the dataset
    :return: Dataset object
    """

    return Dataset(data_path)


def from_default() -> 'Dataset':
    """
    Create a new Dataset object from ../data/subjects
    :return: Dataset object
    """

    data_path = os.path.join(os.getcwd(), '..', 'data', 'VitalCamSet')
    return from_path(data_path)


class Dataset:
    """
    Class to handle the VitalCamSet dataset
    """

    data_path: str

    def __init__(self, data_path: str):
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
            settings = Dataset.get_settings()

        scenarios = []

        for subject in self.get_subjects():
            for setting in settings:
                scenarios.append((subject, setting))

        return scenarios

    def get_video_path(self, subject: str, scenario: str) -> str:
        """
        Get the path to the video file for a given subject and scenario
        :param subject: subject name
        :param scenario: scenario name
        :return: path to the video file
        """

        subject_path = os.path.join(self.data_path, subject)
        scenario_path = os.path.join(subject_path, scenario)
        video_path = os.path.join(scenario_path, 'Logitech HD Pro Webcam C920.avi')

        return video_path

    def get_video_gray(self, subject: str,
                       scenario: str,
                       progress: bool = True) -> tuple[np.ndarray, utils.VideoParams]:
        """
        Get the frames of a given subject and scenario in grayscale
        :param subject: subject name
        :param scenario: scenario name
        :param progress: whether to show progress bar
        :return: numpy array of frames and video parameters
        """

        video_path = self.get_video_path(subject, scenario)
        return utils.read_video_gray(video_path, progress)

    def get_video_bgr(self, subject: str, scenario: str, progress: bool = True) -> tuple[np.ndarray, utils.VideoParams]:
        """
        Get the frames of a given subject and scenario in BGR
        :param subject: subject name
        :param scenario: scenario name
        :param progress: whether to show progress bar
        :return: numpy array of frames and video parameters
        """

        video_path = self.get_video_path(subject, scenario)
        return utils.read_video_bgr(video_path, progress)

    def get_video_rgb(self, subject: str, scenario: str, progress: bool = True) -> tuple[np.ndarray, utils.VideoParams]:
        """
        Get the frames of a given subject and scenario in RGB
        :param subject: subject name
        :param scenario: scenario name
        :param progress: whether to show progress bar
        :return: numpy array of frames and video parameters
        """

        video_path = self.get_video_path(subject, scenario)
        frames, meta = utils.read_video_bgr(video_path, progress)
        return utils.bgr_to_rgb(frames), meta

    def get_unisens_entry(self, subject: str, scenario: str, entry: utils.VitalSigns) -> tuple[np.ndarray, int]:
        """
        Read an entry from an unisens dataset
        :param subject: subject
        :param scenario: scenario
        :param entry: vital signal
        :return: numpy array of the signal and the sampling rate
        """

        subject_path = os.path.join(
            self.data_path,
            subject,
            scenario,
            'synced_Logitech HD Pro Webcam C920')

        return utils.read_unisens_entry(subject_path, entry)

    def get_ground_truth_rr_signal(self, subject: str, scenario: str) -> tuple[np.ndarray, int]:
        """
        Get the ground truth respiratory rate signal for a given subject and scenario
        :param subject:
        :param scenario:
        :return:
        """
        return self.get_unisens_entry(subject, scenario, utils.VitalSigns.thorax_abdomen)

    def contains(self, subject: str, scenario: str) -> bool:
        """
        Check if a given subject and scenario exists in the dataset
        :param subject:
        :param scenario:
        :return:
        """
        subject_path = os.path.join(
            self.data_path,
            subject,
            scenario)

        return os.path.exists(subject_path)
