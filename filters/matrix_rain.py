"""Animated > Matrix Rain."""

import cv2
import numpy as np

from common import FONT
from .base import Filter


MATRIX_FADE = 2.2        # how fast the green trails fade (higher = shorter trails)
GRAY_W = np.array([0.114, 0.587, 0.299], np.float32)   # BGR -> brightness weights


class MatrixRain(Filter):
    """Falling columns of mirrored green characters over a dark green you."""
    name = "Matrix Rain"
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$+-*/=%#&<>?!:"
    CW, CH = 12, 18
    LEVELS = 16          # brightness steps for the fading trails
    # you, as dark green: each output channel = brightness * factor
    BACKGROUND = np.outer([0.08, 0.35, 0.08], GRAY_W).astype(np.float32)
    # rain color from (trail, head): trail is green, the leading char is white
    RAIN = np.array([[0.25, 1.0], [1.0, 1.0], [0.2, 1.0]], np.float32)

    def __init__(self):
        self.rng = np.random.default_rng()
        glyphs = []
        for c in self.CHARS:
            img = np.zeros((self.CH, self.CW), np.uint8)
            cv2.putText(img, c, (1, self.CH - 4), FONT, self.CH / 34, 255, 1, cv2.LINE_AA)
            glyphs.append(cv2.flip(img, 1))    # mirrored, like the movie
        # every glyph pre-rendered at LEVELS brightnesses: (level, char, CH, CW)
        levels = np.linspace(0, 1, self.LEVELS, dtype=np.float32)[:, None, None, None]
        self.atlas = (np.stack(glyphs)[None] * levels).astype(np.uint8)
        self.n_chars = len(glyphs)
        self.grid = None

    def _new_speeds(self, n):
        return self.rng.uniform(6, 22, n)      # rows per second

    def _reset(self, rows, cols):
        self.grid = (rows, cols)
        self.chars = self.rng.integers(0, self.n_chars, (rows, cols))
        self.bright = np.zeros((rows, cols), np.float32)
        self.heads = self.rng.uniform(-rows, rows, cols)
        self.speeds = self._new_speeds(cols)

    def render(self, frame, ctx):
        h, w = frame.shape[:2]
        rows, cols = h // self.CH, w // self.CW
        if self.grid != (rows, cols):
            self._reset(rows, cols)

        # move every column's head down, light up the cells it passed
        self.bright *= np.exp(-MATRIX_FADE * ctx.dt)
        old = np.floor(self.heads).astype(int)
        self.heads += self.speeds * ctx.dt
        new = np.floor(self.heads).astype(int)
        for c in np.nonzero(new > old)[0]:
            r0, r1 = max(old[c] + 1, 0), min(new[c], rows - 1)
            if r1 >= r0:
                self.bright[r0:r1 + 1, c] = 1.0
                self.chars[r0:r1 + 1, c] = self.rng.integers(0, self.n_chars, r1 - r0 + 1)
        done = self.heads > rows * 1.5          # fell off the bottom -> restart at top
        self.heads[done] = self.rng.uniform(-rows * 0.5, 0, done.sum())
        self.speeds[done] = self._new_speeds(done.sum())
        flicker = self.rng.random((rows, cols)) < 0.03
        self.chars[flicker] = self.rng.integers(0, self.n_chars, flicker.sum())

        # draw: glyphs * brightness; the leading character is white
        is_head = np.arange(rows)[:, None] == np.floor(self.heads).astype(int)[None, :]
        def to_img(level):                     # (rows, cols) levels -> glyph image
            tiles = self.atlas[level, self.chars].transpose(0, 2, 1, 3)
            return tiles.reshape(rows * self.CH, cols * self.CW)
        trail = to_img(np.rint(self.bright * (self.LEVELS - 1)).astype(int))
        head = to_img(is_head * (self.LEVELS - 1))
        rain = cv2.transform(cv2.merge([trail, head]), self.RAIN)

        out = cv2.transform(frame, self.BACKGROUND)
        rh, rw = rows * self.CH, cols * self.CW
        out[:rh, :rw] = cv2.add(out[:rh, :rw], rain)
        return out
