"""Photo > Moody B&W: high contrast black & white with grain."""

import cv2
import numpy as np

from .base import Filter, add_grain, apply_vignette, curve


class MoodyBW(Filter):
    """Black & white, strong contrast, crushed shadows, grain, dark corners."""
    name = "Moody B&W"

    def __init__(self):
        self.rng = np.random.default_rng()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        def contrast(x):
            x = np.clip((x - 0.08) / 0.84, 0, 1)             # crush blacks, clip whites
            return x * x * (3 - 2 * x)                       # S-curve = high contrast
        self.tone = curve(contrast)

    def render(self, frame, ctx):
        g = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))   # local contrast
        g = cv2.LUT(g, self.tone)
        g = apply_vignette(g, 0.65)
        g = add_grain(g, 16, self.rng)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
