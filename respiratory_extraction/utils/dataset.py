import os
import re
import numpy as np

from typing import List
from . import read_video_gray, read_video_bgr, VideoParams
from .unisens import read_unisens_entry

import respiratory_extraction.models.baseline as baseline


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
    def get_scenarios() -> List[str]:
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

    def read_video_gray(self, subject: str, scenario: str) -> tuple[np.array, VideoParams]:
        """
        Get the frames of a given subject and scenario in grayscale
        :param subject: subject name
        :param scenario: scenario name
        :return: numpy array of frames and video parameters
        """

        video_path = self.get_video_path(subject, scenario)
        return read_video_gray(video_path)

    def read_video_bgr(self, subject: str, scenario: str) -> tuple[np.array, VideoParams]:
        """
        Get the frames of a given subject and scenario in BGR
        :param subject: subject name
        :param scenario: scenario name
        :return: numpy array of frames and video parameters
        """

        video_path = self.get_video_path(subject, scenario)
        return read_video_bgr(video_path)

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

    def get_ground_truth_rr(self, subject: str, scenario: str) -> float:
        """
        Get the ground truth respiratory rate for a given subject and scenario
        :param subject: subject name
        :param scenario: scenario name
        :return: ground truth respiratory rate in Hz
        """

        gt_signal, gt_sample_rate = self.read_unisens_entry(subject, scenario, '3_Thorax')
        gt_fft, gt_freq = baseline.calculate_fft(gt_signal.tolist(), gt_sample_rate)
        gt_max_freq, _ = baseline.calculate_respiratory_rate(gt_fft, gt_freq)
        return gt_max_freq
