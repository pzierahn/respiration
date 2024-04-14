import os
import re
from typing import List


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
