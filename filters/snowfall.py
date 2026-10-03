"""Animated > Snowfall."""

import cv2
import numpy as np

from .base import Filter, curve3


SNOW_FLAKES = 350


class Snowfall(Filter):
    """Snowflakes drifting down over a cool, slightly hazy camera."""
    name = "Snowfall"

    def __init__(self, n=SNOW_FLAKES):
        self.rng = np.random.default_rng()
        self.x = self.rng.uniform(0, 1, n)
        self.y = self.rng.uniform(0, 1, n)
        self.size = self.rng.uniform(1, 4.5, n)          # pixels at 720p
        self.speed = 0.03 + self.size * 0.035              # big flakes are closer = faster
        self.phase = self.rng.uniform(0, 2 * np.pi, n)
        self.sway = self.rng.uniform(0.005, 0.03, n)
        haze = 16 / 255
        self.cool = curve3(lambda x: x * 1.06 * 0.9 + haze,   # bluer
                           lambda x: x * 0.9 + haze,
                           lambda x: x * 0.92 * 0.9 + haze)   # less red

    def render(self, frame, ctx):
        h, w = frame.shape[:2]
        self.y += self.speed * ctx.dt
        self.x = (self.x + np.sin(ctx.t * 1.3 + self.phase) * self.sway * ctx.dt) % 1.0
        gone = self.y > 1.03
        self.y[gone] = -0.03
        self.x[gone] = self.rng.uniform(0, 1, gone.sum())

        layer = np.zeros((h, w), np.uint8)
        k = h / 720
        for x, y, s in zip(self.x, self.y, self.size):
            cv2.circle(layer, (int(x * w), int(y * h)), max(1, int(s * k)), 255, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.2)
        return cv2.add(cv2.LUT(frame, self.cool), cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR))
