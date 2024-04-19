import json

import cv2
import numpy as np


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


def read_video_bgr(path: str) -> tuple[np.array, VideoParams]:
    """
    Read a video file and return a numpy array of frames in BGR format
    :param path: path to the video file
    :return: numpy array of frames and video parameters
    """

    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    params = VideoParams(frame_count, fps)

    frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()

    return np.array(frames), params


def read_video_gray(path: str) -> tuple[np.array, VideoParams]:
    """
    Read a video file and return a numpy array of frames in grayscale
    :param path: path to the video file
    :return: numpy array of frames and video parameters
    """

    frames, params = read_video_bgr(path)
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    return np.array(frames), params
