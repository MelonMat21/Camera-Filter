"""Photo > Dreamy Glow: soft bloom on bright areas."""

import cv2
import numpy as np

from .base import Filter, curve3, to_uint8


class DreamyGlow(Filter):
    """Bright areas bleed a soft glow (bloom), everything slightly soft."""
    name = "Dreamy Glow"

    def __init__(self):
        self.haze = curve3(lambda x: x * 0.96 + 0.04,          # rosy, lifted haze
                           lambda x: x * 0.98 * 0.96 + 0.04,
                           lambda x: x * 1.04 * 0.96 + 0.04)

    def render(self, frame, ctx):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        f = small.astype(np.float32) / 255                    # small image: float is cheap
        bright = f * np.clip((f.max(axis=2) - 0.55) / 0.45, 0, 1)[..., None]   # only bright parts
        glow = to_uint8(cv2.GaussianBlur(bright, (0, 0), 6) * 1.3 * 255)
        soft = cv2.GaussianBlur(small, (0, 0), 1.5)
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
        soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)

        base = cv2.addWeighted(frame, 0.75, soft, 0.25, 0)    # gentle softness
        # "screen" blend: 1 - (1-base)(1-glow) -> brightens, never darkens
        screen = cv2.bitwise_not(cv2.multiply(cv2.bitwise_not(base), cv2.bitwise_not(glow),
                                              scale=1 / 255))
        return cv2.LUT(screen, self.haze)
