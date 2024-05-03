from typing import Optional

import cv2
import json
import numpy as np
from tqdm.auto import tqdm


class VideoParams:
    """
    Class to store video parameters such as frame count and fps
    """
    start_position: int
    num_frames: int
    fps: int

    def __init__(self, start_position: int, num_frames: int, fps: int):
        """
        Initialize the video parameters
        :param start_position:
        :param num_frames: number of frames in the video
        :param fps: frames per second
        """
        self.start_position = start_position
        self.num_frames = num_frames
        self.fps = fps

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)


def get_frame_count(path: str) -> int:
    """
    Get the number of frames in a video file
    :param path: path to the video file
    :return: number of frames in the video
    """

    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count


def read_video_bgr(path: str,
                   num_frames: Optional[int] = None,
                   start_position: int = 0,
                   show_progress: bool = False) -> tuple[np.array, VideoParams]:
    """
    Read a video file and return a numpy array of frames in BGR format
    :param path: path to the video file
    :param num_frames: number of frames to read
    :param start_position: starting frame index
    :param show_progress: whether to show progress bar
    :return: numpy array of frames and video parameters
    """

    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the number of frames to read
    if num_frames is None:
        num_frames = frame_count - start_position

    # Check if the start position and number of frames are valid
    assert start_position + num_frames <= frame_count, \
        f"Invalid start position {start_position} and num frames {num_frames}"

    # Get the frames per second
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Seek to the start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_position)

    frames = []
    for _ in tqdm(range(num_frames), disable=(not show_progress)):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()

    # Create video parameters object
    params = VideoParams(start_position, num_frames, fps)
    return np.array(frames), params


def convert_to_gray(frames: np.array) -> np.array:
    """
    Convert a numpy array of frames from BGR to grayscale
    :param frames: numpy array of frames in BGR format
    :return: numpy array of frames in grayscale
    """

    return np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])


def read_video_gray(path: str,
                    num_frames: Optional[int] = None,
                    start_position: int = 0,
                    show_progress: bool = True) -> tuple[np.array, VideoParams]:
    """
    Read a video file and return a numpy array of frames in grayscale
    :param path: path to the video file
    :param num_frames: number of frames to read
    :param start_position: starting frame index
    :param show_progress: whether to show progress bar
    :return: numpy array of frames and video parameters
    """

    frames, params = read_video_bgr(path, num_frames, start_position, show_progress)
    return convert_to_gray(frames), params


def down_sample_video(frames: np.array, dim: int = 36) -> np.array:
    """
    Down sample a numpy array of frames to a target dimension
    :param frames: numpy array of frames
    :param dim: target dimension
    :return: down sampled numpy array of frames
    """

    return np.array([cv2.resize(frame, (dim, dim), interpolation=cv2.INTER_AREA) for frame in frames])


def bgr_to_rgb(frames: np.array) -> np.array:
    """
    Convert a numpy array of frames from BGR to RGB
    :param frames: numpy array of frames in BGR format
    :return: numpy array of frames in RGB format
    """

    return np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
