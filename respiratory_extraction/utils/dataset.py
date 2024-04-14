import os
import re
import numpy as np

from typing import List
from . import read_video_gray, VideoParams
from .unisens import read_unisens_entry


class Dataset:
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
    def get_scenarios():
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

    def get_video_path(self, subject: str, scenario) -> str:
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

    def read_video_gray(self, subject: str, scenario: str) -> tuple[np.array, VideoParams]:
        """
        Read a video file and return a numpy array of frames
        :param subject: subject name
        :param scenario: scenario name
        :return: numpy array of frames and video parameters
        """

        video_path = self.get_video_path(subject, scenario)
        return read_video_gray(video_path)

    def read_unisens_entry(self, subject: str, scenario: str, entry: str) -> tuple[np.ndarray, int]:
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

        return read_unisens_entry(subject_path, entry)
