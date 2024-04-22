import json

import cv2
import numpy as np
from tqdm.auto import tqdm


class VideoParams:
    """
    Class to store video parameters such as frame count and fps
    """
    frame_count: int
    fps: int

    def __init__(self, frame_count: int, fps: int):
        self.frame_count = frame_count
        self.fps = fps

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)


def read_video_bgr(path: str, progress: bool = True) -> tuple[np.array, VideoParams]:
    """
    Read a video file and return a numpy array of frames in BGR format
    :param path: path to the video file
    :param progress: whether to show progress bar
    :return: numpy array of frames and video parameters
    """

    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    params = VideoParams(frame_count, fps)

    frames = []
    for _ in tqdm(range(frame_count), disable=(not progress)):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()

    return np.array(frames), params


def convert_to_gray(frames: np.array) -> np.array:
    """
    Convert a numpy array of frames from BGR to grayscale
    :param frames: numpy array of frames in BGR format
    :return: numpy array of frames in grayscale
    """

    return np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])


def read_video_gray(path: str, progress: bool = True) -> tuple[np.array, VideoParams]:
    """
    Read a video file and return a numpy array of frames in grayscale
    :param path: path to the video file
    :param progress: whether to show progress bar
    :return: numpy array of frames and video parameters
    """

    frames, params = read_video_bgr(path, progress=progress)
    return convert_to_gray(frames), params


def down_sample_video(frames: np.ndarray, dim: int = 36) -> np.array:
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
