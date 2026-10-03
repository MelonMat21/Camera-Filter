"""Cartoons > Pencil Sketch."""

import cv2

from .base import Filter, curve3, half


class PencilSketch(Filter):
    """Grey pencil drawing on paper ("color dodge" trick)."""
    name = "Pencil Sketch"

    def __init__(self):
        # darker, bolder strokes (x^1.6) on warm off-white paper
        self.paper = curve3(lambda x: x ** 1.6 * 0.93,
                            lambda x: x ** 1.6 * 0.95,
                            lambda x: x ** 1.6 * 0.97)

    def render(self, frame, ctx):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(255 - half(gray), (0, 0), 4)   # blur the negative
        blur = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
        sketch = cv2.divide(gray, 255 - blur, scale=256)        # edges -> dark strokes
        return cv2.LUT(cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR), self.paper)
