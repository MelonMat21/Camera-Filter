"""
ascii_cam.py - Real-time webcam as grey terminal ASCII, with a finger-shape window

Install:   pip install opencv-contrib-python numpy mediapipe
Run:       python ascii_cam.py

How to use:
    1. Landing: touch thumb + index finger together on BOTH hands.
    2. Pull your fingers apart - the 4 fingertips make a shape.
       Everything is ASCII; inside the shape you see the filter
       (for now, the normal camera).
       The shape follows your fingers. Drop a hand out of view to reset.

Needs hand_landmarker.task (MediaPipe hand model) in the same folder.

Controls:
    q or ESC : quit
    + / -    : more / fewer characters (more = more detail, slower)
    c        : toggle grey <-> bright white
    i        : invert brightness (dense characters for dark areas)
    s        : save a screenshot (ascii_<timestamp>.png)
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------- CONFIG ---------------------------------------
CAMERA_INDEX = 0          # 0 = default webcam. Try 1, 2 if you have several.
CAMERA_W, CAMERA_H = 1280, 720   # higher res = more accurate hand tracking
START_COLS = 120          # how many characters wide the ASCII image is
CELL_W, CELL_H = 9, 16    # pixel size of one character cell (terminal-like 9x16)
CANDIDATE_CHARS = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
GREY = (192, 192, 192)    # BGR text color (classic terminal grey)
# ----------------------------------------------------------------------------


def build_glyph_atlas(chars, cell_w, cell_h):
    """
    Pre-render every character ONCE into a small image, then sort the
    characters from least "ink" to most "ink". Returns an array of shape
    (num_chars, cell_h, cell_w).

    Sorting by measured ink means the brightness ramp is accurate for
    whatever font OpenCV draws, instead of trusting a hand-made ramp.
    """
    scale = cell_h / 32.0
    glyphs = []
    for c in chars:
        img = np.zeros((cell_h, cell_w), np.uint8)
        cv2.putText(img, c, (0, cell_h - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, 255, 1, cv2.LINE_AA)
        glyphs.append(img)

    density = np.array([g.mean() for g in glyphs])
    order = np.argsort(density, kind="stable")

    # Remove near-duplicate densities so the ramp has distinct steps
    sorted_glyphs, last = [], -1.0
    for i in order:
        if density[i] - last > 0.35 or not sorted_glyphs:
            sorted_glyphs.append(glyphs[i])
            last = density[i]
    return np.stack(sorted_glyphs)


def frame_to_ascii(gray, cols, atlas, invert=False):
    """
    Core algorithm. Converts a grayscale frame into an ASCII-art image.

    1. Compute how many rows we need (characters are taller than wide).
    2. Shrink the frame so 1 pixel == 1 character cell.
    3. Stretch contrast to the full 0-255 range (looks much better).
    4. Map each pixel's brightness to a character index.
    5. Look up each character's pre-rendered glyph and tile them together.
    """
    cell_h, cell_w = atlas.shape[1], atlas.shape[2]
    h, w = gray.shape

    # 1. rows so the output keeps the camera's aspect ratio
    rows = max(1, int(h / w * cols * cell_w / cell_h))

    # 2. downscale (INTER_AREA averages pixels = best for shrinking)
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

    # 3. auto-contrast
    small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX)

    # 4. brightness (0..255) -> index (0..n-1)
    n = len(atlas)
    idx = (small.astype(np.int32) * (n - 1)) // 255
    if invert:
        idx = (n - 1) - idx

    # 5. gather glyphs: (rows, cols, cell_h, cell_w) -> one big image
    tiles = atlas[idx]
    img = tiles.transpose(0, 2, 1, 3).reshape(rows * cell_h, cols * cell_w)
    return img


def colorize(mono, grey=True):
    """Turn the 1-channel glyph image into a BGR image (grey or white)."""
    if not grey:
        return cv2.merge([mono, mono, mono])
    b = (mono * (GREY[0] / 255)).astype(np.uint8)
    g = (mono * (GREY[1] / 255)).astype(np.uint8)
    r = (mono * (GREY[2] / 255)).astype(np.uint8)
    return cv2.merge([b, g, r])


# ----------------------------- HAND TRACKING --------------------------------
THUMB_TIP, INDEX_TIP = 4, 8      # MediaPipe landmark ids
WRIST, MIDDLE_BASE = 0, 9        # used to measure hand size
PINCH_ON, PINCH_OFF = 0.25, 0.35 # pinch distance / hand size (two thresholds
                                 # so it doesn't flicker at the edge)
# Smoothing (One Euro filter): steady when still, no lag when moving fast
SMOOTH_MIN_CUTOFF = 1.0          # lower = less jitter when hands are still
SMOOTH_BETA = 0.02               # higher = less lag when hands move fast
SHAPE_SCALE = 1.4                # grow the shape out from its center (1.0 = exactly at fingertips)
DETECT_CONFIDENCE = 0.7          # 0..1, higher = fewer false/wobbly detections
LOST_GRACE = 0.6                 # seconds a hand can vanish before the shape resets


def make_hand_detector():
    """Load MediaPipe's hand model (hand_landmarker.task next to this file)."""
    model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "hand_landmarker.task")
    if not os.path.exists(model):
        raise SystemExit("Missing hand_landmarker.task - download it from\n"
                         "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                         "hand_landmarker/float16/latest/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=DETECT_CONFIDENCE,
        min_hand_presence_confidence=DETECT_CONFIDENCE,
        min_tracking_confidence=DETECT_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def pinch_ratio(hand):
    """Thumb-tip to index-tip distance, divided by the hand's size.
    Dividing by hand size makes it work whether you're near or far."""
    def dist(a, b):
        return np.hypot(hand[a].x - hand[b].x, hand[a].y - hand[b].y)
    return dist(THUMB_TIP, INDEX_TIP) / max(dist(WRIST, MIDDLE_BASE), 1e-6)


class OneEuroFilter:
    """Adaptive smoothing: heavy smoothing when the points barely move
    (kills jitter), light smoothing when they move fast (kills lag)."""

    def __init__(self, min_cutoff=SMOOTH_MIN_CUTOFF, beta=SMOOTH_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = self.dx + a_d * ((x - self.x) / dt - self.dx)
        speed = np.abs(self.dx)              # per-coordinate speed (pixels/sec)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = self.x + a * (x - self.x)
        self.t = t
        return self.x


def sort_hands(hands):
    """Left-most hand first, so each hand keeps the same slot every frame."""
    return sorted(hands, key=lambda hand: hand[WRIST].x)


def expand_shape(corners, scale=SHAPE_SCALE):
    """Push every corner away from the shape's center to make it bigger."""
    center = corners.mean(axis=0)
    return center + (corners - center) * scale


def fingertips(hands, w, h):
    """The 4 corners of the shape, in pixels, going around the outside:
    left index -> right index -> right thumb -> left thumb."""
    left, right = sort_hands(hands)
    def px(hand, i):
        return (hand[i].x * w, hand[i].y * h)
    return np.array([px(left, INDEX_TIP), px(right, INDEX_TIP),
                     px(right, THUMB_TIP), px(left, THUMB_TIP)], np.float32)


def draw_fingers(img, hands, pinched):
    """Dot on thumb + index tip of each hand, line between them.
    Grey = apart, white = touching."""
    h, w = img.shape[:2]
    for hand, is_pinched in zip(hands, pinched):
        color = (255, 255, 255) if is_pinched else GREY
        pts = [(int(hand[i].x * w), int(hand[i].y * h)) for i in (THUMB_TIP, INDEX_TIP)]
        cv2.line(img, pts[0], pts[1], color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(img, pt, 8, color, -1, cv2.LINE_AA)


def full_ascii(frame, cols, atlas, invert, grey):
    """The whole frame as ASCII, resized back to the camera's size."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ascii_img = colorize(frame_to_ascii(gray, cols, atlas, invert), grey)
    return cv2.resize(ascii_img, (w, h), interpolation=cv2.INTER_AREA)


def apply_filter(frame):
    """The filter shown inside the finger shape. For now: the normal camera.
    Swap this out later for other filters."""
    return frame


def filter_in_shape(frame, ascii_img, corners):
    """ASCII everywhere, the filter only inside the finger shape."""
    h, w = frame.shape[:2]
    filtered = apply_filter(frame)

    # convexHull keeps the shape from twisting into a bow-tie if fingers cross
    hull = cv2.convexHull(expand_shape(corners).astype(np.int32))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_AA)

    out = ascii_img.copy()
    out[mask > 0] = filtered[mask > 0]
    cv2.polylines(out, [hull], True, GREY, 2, cv2.LINE_AA)
    for x, y in corners:
        cv2.circle(out, (int(x), int(y)), 6, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}. "
                         "Check permissions or try another CAMERA_INDEX.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    print("Camera resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    atlas = build_glyph_atlas(CANDIDATE_CHARS, CELL_W, CELL_H)
    print(f"Using {len(atlas)} distinct brightness levels.")
    detector = make_hand_detector()

    cols, grey, invert = START_COLS, True, False
    prev, fps = time.time(), 0.0
    start = time.time()
    pinch_state = [False, False]     # per detected hand, for hysteresis
    landed = False                   # True after the double-pinch "landing"
    corners = None                   # smoothed fingertip positions
    smoother = OneEuroFilter()
    last_seen = 0.0                  # last time both hands were visible

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not received, stopping.")
            break

        frame = cv2.flip(frame, 1)                       # mirror, like a selfie
        h, w = frame.shape[:2]

        # --- detect hands (MediaPipe wants RGB + increasing timestamps) ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, int((time.time() - start) * 1000))
        hands = sort_hands(result.hand_landmarks)

        pinched = []
        for i, hand in enumerate(hands):
            r = pinch_ratio(hand)
            was = pinch_state[i] if i < len(pinch_state) else False
            pinched.append(r < (PINCH_OFF if was else PINCH_ON))
        pinch_state = pinched + [False] * (2 - len(pinched))

        now = time.time()
        if len(hands) == 2:
            last_seen = now
            # Landing: touch thumb + index on BOTH hands to start the shape
            if all(pinched):
                landed = True
            corners = smoother(fingertips(hands, w, h), now)
        elif now - last_seen > LOST_GRACE:
            # a hand left the camera for too long -> back to waiting
            landed, corners = False, None
            smoother = OneEuroFilter()

        ascii_img = full_ascii(frame, cols, atlas, invert, grey)   # landing screen
        if landed and corners is not None:
            out = filter_in_shape(frame, ascii_img, corners)
        else:
            out = ascii_img
            draw_fingers(out, hands, pinched)

        # FPS counter (smoothed)
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        status = "filter shape" if landed else f"pinch both hands to start | hands {len(hands)}/2"
        cv2.putText(out, f"{fps:4.1f} fps | {status}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)

        cv2.imshow("ASCII Cam", out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("+"), ord("=")):
            cols = min(300, cols + 10)
        elif key in (ord("-"), ord("_")):
            cols = max(20, cols - 10)
        elif key == ord("c"):
            grey = not grey
        elif key == ord("i"):
            invert = not invert
        elif key == ord("s"):
            name = time.strftime("ascii_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, out)
            print("Saved", name)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()