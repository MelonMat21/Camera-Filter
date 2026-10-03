"""Things shared by the main app and the filters."""

import os
import urllib.request

import cv2

GREY = (192, 192, 192)    # BGR text color (classic terminal grey)
FONT = cv2.FONT_HERSHEY_SIMPLEX

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                 "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def model_path(name):
    """Path to a MediaPipe model next to this file; downloads it if missing."""
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print(f"Downloading {name} ...")
        try:
            urllib.request.urlretrieve(MODEL_URLS[name], path)
        except OSError as e:
            if os.path.exists(path):
                os.remove(path)
            raise SystemExit(f"Could not download {name} ({e}). Get it from\n{MODEL_URLS[name]}")
    return path
