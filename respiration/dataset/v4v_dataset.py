import os
import numpy as np
import respiration.utils as utils

from typing import Optional


class V4VDataset:
    """
    Class to handle the V4V dataset.
    """

    # Path to the V4V dataset
    data_path: str

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            # Use the default path if not provided
            self.data_path = utils.dir_path('data', 'V4V-Dataset')
        else:
            self.data_path = data_path

    def list_videos(self) -> list[str]:
        """
        List all the videos in the dataset.
        """
        path = os.path.join(self.data_path, 'Phase 1_ Training_Validation sets', 'Videos')

        # list all mkv files in the directory
        videos = [f for f in os.listdir(path) if f.endswith('.mkv')]

        return videos

    def read_ground_truths(self) -> str:
        """
        Read the ground truth files.
        """

        parts = []

        # Read the template file
        path = os.path.join(self.data_path, 'Phase 1_ Training_Validation sets', 'codalab', 'template.txt')

        with open(path, 'r') as file:
            parts.append(file.read().strip())

        # Read the ground truth files
        directory = os.path.join(self.data_path, 'Phase 1_ Training_Validation sets', 'Ground truth', 'Physiology')

        # list all csv files in the directory
        for file in os.listdir(directory):
            if not file.endswith('.txt'):
                continue

            with open(os.path.join(directory, file), 'r') as f:
                parts.append(f.read().strip())

        return '\n'.join(parts)

    def get_metadata(self):
        """
        Parse the data.
        """
        text = self.read_ground_truths()

        # Split the text into parts
        lines = text.split('\n')

        data = []

        for line in lines:
            parts = line.split(', ')
            video, vital = parts[0], parts[1]
            signal = np.array([float(x) for x in parts[2:]])

            subject = video.split('_')[0]
            setting = video.split('_')[1].removesuffix('.mkv')

            data.append({
                'subject': subject,
                'setting': setting,
                'video': video,
                'vital': vital,
                'signal': signal,
            })

        return data

    def get_video_rgb(self, video: str) -> tuple[np.array, utils.VideoParams]:
        """Get the RGB frames of a video."""
        path = os.path.join(self.data_path, 'Phase 1_ Training_Validation sets', 'Videos', video)
        return utils.read_video_rgb(path)
