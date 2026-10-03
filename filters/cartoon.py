"""Cartoons > Cartoon: flat colors and thick outlines."""

import cv2
import numpy as np

from .base import Filter, half, posterize_curve


class Cartoon(Filter):
    """Flat colors (smooth + posterize) with thick black outlines."""
    name = "Cartoon"

    def __init__(self):
        self.flat = posterize_curve(32)

    def render(self, frame, ctx):
        h, w = frame.shape[:2]
        small = half(frame)
        color = small
        for _ in range(3):                                     # smooth, keep edges
            color = cv2.bilateralFilter(color, 5, 50, 5)
        color = cv2.LUT(color, self.flat)                      # flat color bands

        gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 5)   # 0 = outline
        edges = cv2.erode(edges, np.ones((2, 2), np.uint8))      # thicker lines
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cv2.resize(cartoon, (w, h), interpolation=cv2.INTER_LINEAR)
