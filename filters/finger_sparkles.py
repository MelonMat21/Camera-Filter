"""Animated > Finger Sparkles."""

import cv2
import numpy as np

from .base import Filter, half


class FingerSparkles(Filter):
    """Normal camera; glowing sparkles fly off every fingertip (more when you move)."""
    name = "Finger Sparkles"
    TIPS = (4, 8, 12, 16, 20)
    COLORS = np.array([(255, 255, 255), (80, 215, 255), (230, 150, 255),
                       (255, 230, 120), (150, 255, 200)], np.float32)   # BGR
    RATE = 25            # sparkles per second per fingertip when still
    MAX = 1500

    def __init__(self):
        self.rng = np.random.default_rng()
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)
        self.max_life = np.zeros(0, np.float32)
        self.color = np.zeros(0, int)
        self.size = np.zeros(0, np.float32)
        self.phase = np.zeros(0, np.float32)
        self.prev_tips = None

    def _emit(self, tips, ctx, scale):
        moved = np.zeros(len(tips))
        if self.prev_tips is not None and self.prev_tips.shape == tips.shape:
            moved = np.linalg.norm(tips - self.prev_tips, axis=1)
        self.prev_tips = tips
        counts = self.rng.poisson(self.RATE * ctx.dt + moved * 0.08)
        n = int(counts.sum())
        if n == 0:
            return
        angle = self.rng.uniform(0, 2 * np.pi, n)
        speed = self.rng.uniform(40, 220, n) * scale
        life = self.rng.uniform(0.35, 0.9, n)
        self.pos = np.vstack([self.pos, np.repeat(tips, counts, axis=0) + self.rng.normal(0, 3, (n, 2))])
        self.vel = np.vstack([self.vel, np.stack([np.cos(angle), np.sin(angle)], 1) * speed[:, None]])
        self.life = np.concatenate([self.life, life])
        self.max_life = np.concatenate([self.max_life, life])
        self.color = np.concatenate([self.color, self.rng.integers(0, len(self.COLORS), n)])
        self.size = np.concatenate([self.size, self.rng.uniform(1.5, 4.5, n) * scale])
        self.phase = np.concatenate([self.phase, self.rng.uniform(0, 2 * np.pi, n)])

    def overlay(self, out, ctx):
        h, w = out.shape[:2]
        scale = h / 720
        tips = np.array([[hand[i].x * w, hand[i].y * h] for hand in ctx.hands for i in self.TIPS],
                        np.float32).reshape(-1, 2)
        if len(tips):
            self._emit(tips, ctx, scale)
        else:
            self.prev_tips = None

        # physics: move, slow down, fall a little, age
        self.pos += self.vel * ctx.dt
        self.vel *= np.exp(-2.5 * ctx.dt)
        self.vel[:, 1] += 120 * scale * ctx.dt
        self.life -= ctx.dt
        keep = self.life > 0
        alive = np.nonzero(keep)[0]
        if len(alive) > self.MAX:                 # too many: drop the oldest
            keep[alive[:-self.MAX]] = False
        for name in ("pos", "vel", "life", "max_life", "color", "size", "phase"):
            setattr(self, name, getattr(self, name)[keep])
        if len(self.life) == 0:
            return

        # draw each sparkle as a twinkling 4-point star, then add a soft glow
        layer = np.zeros_like(out)
        fade = self.life / self.max_life
        twinkle = 0.55 + 0.45 * np.sin(ctx.t * 25 + self.phase)
        for (x, y), a, tw, ci, s in zip(self.pos, fade, twinkle, self.color, self.size):
            x, y = int(x), int(y)
            col = tuple(float(v) for v in self.COLORS[ci] * a)
            arm = max(1, int(s * 2.5 * tw))
            cv2.line(layer, (x - arm, y), (x + arm, y), col, 1, cv2.LINE_AA)
            cv2.line(layer, (x, y - arm), (x, y + arm), col, 1, cv2.LINE_AA)
            cv2.circle(layer, (x, y), max(1, int(s * 0.6 * tw)), col, -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(half(layer), (0, 0), 2.5 * scale)   # blur small = fast
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
        out[:] = cv2.add(cv2.add(out, layer), cv2.add(glow, glow))
