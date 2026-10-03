"""Cartoons > Comic Book: bold colors, ink lines, halftone dots."""

import cv2
import numpy as np

from .base import Filter, _cache, curve, half, posterize_curve


class ComicBook(Filter):
    """Bold saturated colors, black ink lines, halftone dots in the shadows."""
    name = "Comic Book"
    DOT = 6                                                    # halftone cell (half-res px)

    def __init__(self):
        self.saturate = curve(lambda x: x * 1.7)
        self.brighten = curve(lambda x: x * 1.1)
        self.flat = posterize_curve(64)

    def render(self, frame, ctx):
        h, w = frame.shape[:2]
        small = half(frame)
        sh, sw = small.shape[:2]

        color = cv2.bilateralFilter(small, 5, 50, 5)
        hue, sat, val = cv2.split(cv2.cvtColor(color, cv2.COLOR_BGR2HSV))
        color = cv2.cvtColor(cv2.merge([hue, cv2.LUT(sat, self.saturate), cv2.LUT(val, self.brighten)]),
                             cv2.COLOR_HSV2BGR)                 # bold colors
        color = cv2.LUT(color, self.flat)

        # halftone: one dot per cell, bigger dot where the image is darker
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        c = self.DOT
        cells = cv2.resize(gray, (-(-sw // c), -(-sh // c)), interpolation=cv2.INTER_AREA)
        dark = np.clip((1 - cells.astype(np.float32) / 255 - 0.2) / 0.8, 0, 1)
        radius = (np.sqrt(dark) * c * 0.6).repeat(c, 0).repeat(c, 1)[:sh, :sw]
        dots = (self._cell_dist(sh, sw) < radius).astype(np.uint8) * 255
        color = cv2.copyTo(cv2.convertScaleAbs(color, alpha=0.45), dots, color)

        # ink lines
        edges = cv2.Canny(cv2.medianBlur(gray, 5), 50, 120)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        color = cv2.bitwise_and(color, color, mask=cv2.bitwise_not(edges))
        return cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)

    def _cell_dist(self, sh, sw):
        """Distance of every pixel from the center of its halftone cell."""
        key = ("halftone", sh, sw, self.DOT)
        if key not in _cache:
            yy, xx = np.indices((sh, sw), np.float32)
            mid = (self.DOT - 1) / 2
            _cache[key] = np.hypot(yy % self.DOT - mid, xx % self.DOT - mid)
        return _cache[key]
