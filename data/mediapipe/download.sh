#!/bin/sh

wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
wget https://raw.githubusercontent.com/apple2373/mediapipe-facemesh/main/data/uv_map.json
