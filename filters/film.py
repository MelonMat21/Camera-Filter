"""Photo > Film: warm tint, grain, vignette."""

import cv2
import numpy as np

from .base import Filter, add_grain, apply_vignette, curve3


class Film(Filter):
    """Warm tint, faded blacks, moving grain, dark corners."""
    name = "Film"

    def __init__(self):
        self.rng = np.random.default_rng()
        fade = 18 / 255                                      # lifted (faded) blacks
        self.tone = curve3(lambda x: x * 0.86 * 0.88 + fade,   # less blue
                           lambda x: x * 0.88 + fade,
                           lambda x: x * 1.12 * 0.88 + fade)   # more red = warm

    def render(self, frame, ctx):
        img = cv2.LUT(frame, self.tone)
        img = apply_vignette(img, 0.55)
        return add_grain(img, 12, self.rng)
