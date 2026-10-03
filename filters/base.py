"""Shared filter helpers + the Filter base class.

Filters use OpenCV calls + lookup tables (LUTs) instead of numpy float math
on the full frame - on 1280x720 that is about 10x faster.
"""

import cv2
import numpy as np

_cache = {}
GRAIN_PAD = 64


def curve(fn):
    """256-entry lookup table from a function on 0..1 values.
    cv2.LUT then applies it to every pixel almost for free."""
    x = np.arange(256, dtype=np.float32) / 255
    return np.clip(fn(x) * 255 + 0.5, 0, 255).astype(np.uint8)


def curve3(fb, fg, fr):
    """One lookup table per channel (B, G, R) for color images."""
    return np.dstack([curve(fb), curve(fg), curve(fr)])      # shape (1, 256, 3)


def posterize_curve(step):
    """Lookup table that flattens colors into bands of `step` (fewer colors)."""
    x = np.arange(256)
    return np.clip(x // step * step + step // 2, 0, 255).astype(np.uint8)


def apply_vignette(img, strength):
    """Darken the corners: 1.0 in the center, 1-strength in the corners."""
    h, w = img.shape[:2]
    key = ("vignette", h, w, img.ndim, strength)
    if key not in _cache:
        y, x = np.ogrid[:h, :w]
        d2 = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
        mask = np.clip((1.0 - strength * d2 / 2.0) * 255, 0, 255).astype(np.uint8)
        _cache[key] = cv2.merge([mask] * 3) if img.ndim == 3 else mask
    return cv2.multiply(img, _cache[key], scale=1 / 255)


def add_grain(img, amount, rng):
    """Film grain. One big noise image is made once, then a random window
    of it is used each frame, so it's fast but still moves."""
    h, w = img.shape[:2]
    key = ("grain", h, w, img.ndim, amount)
    if key not in _cache:
        n = np.random.default_rng(1).normal(0, amount, (h + GRAIN_PAD, w + GRAIN_PAD))
        pos = np.clip(n, 0, 255).astype(np.uint8)
        neg = np.clip(-n, 0, 255).astype(np.uint8)
        if img.ndim == 3:
            pos, neg = cv2.merge([pos] * 3), cv2.merge([neg] * 3)
        _cache[key] = (pos, neg)
    pos, neg = _cache[key]
    oy, ox = (int(v) for v in rng.integers(0, GRAIN_PAD, 2))
    img = cv2.add(img, pos[oy:oy + h, ox:ox + w])
    return cv2.subtract(img, neg[oy:oy + h, ox:ox + w])


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def half(frame):
    """Half-size copy - heavy filters run on this, then scale back up (faster)."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def hroll(img, dx):
    """Shift an image sideways by dx pixels, wrapping around."""
    dx %= img.shape[1]
    if dx == 0:
        return img.copy()
    return np.concatenate([img[:, -dx:], img[:, :-dx]], axis=1)


# Every filter has:
#   render(frame, ctx)  -> image shown inside the finger shape (same size as frame)
#   overlay(out, ctx)   -> optional drawing on top of the whole screen
# ctx has: t (seconds), dt (seconds since last frame), hands (landmarks),
#          mp_image + timestamp_ms (for extra MediaPipe models).
class Filter:
    name = "?"

    def render(self, frame, ctx):
        return frame

    def overlay(self, out, ctx):
        pass
