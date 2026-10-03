"""Animated > Glitch: RGB split and jumping bands."""

import cv2
import numpy as np

from .base import Filter, hroll


class Glitch(Filter):
    """RGB channels split apart + horizontal bands that jump sideways."""
    name = "Glitch"

    def __init__(self):
        self.rng = np.random.default_rng()
        self.next_change, self.shift, self.bands = 0.0, 8, []

    def render(self, frame, ctx):
        h, w = frame.shape[:2]
        if ctx.t >= self.next_change:          # new random glitch every 50-250 ms
            self.next_change = ctx.t + self.rng.uniform(0.05, 0.25)
            self.shift = int(self.rng.uniform(4, 14) * w / 1280) * (3 if self.rng.random() < 0.15 else 1)
            self.bands = []
            for _ in range(self.rng.integers(1, 6)):
                y0 = int(self.rng.integers(0, h))
                bh = int(self.rng.integers(h // 80 + 1, max(h // 10, h // 80 + 2)))
                dx = int(self.rng.integers(-w // 8, w // 8 + 1))
                self.bands.append((y0, min(h, y0 + bh), dx))

        b, g, r = cv2.split(frame)
        out = cv2.merge([hroll(b, -self.shift), g, hroll(r, self.shift)])   # RGB split
        for y0, y1, dx in self.bands:
            out[y0:y1] = hroll(out[y0:y1], dx)
        out[::3] = cv2.convertScaleAbs(out[::3], alpha=0.75)                # scanlines
        return out
