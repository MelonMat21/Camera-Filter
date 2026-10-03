"""
Filter.py - Real-time webcam as grey terminal ASCII, with a finger-shape window
that shows the filter you pick from the navbar.

Install:   pip install opencv-contrib-python numpy mediapipe
Run:       python Filter.py

How to use:
    1. Landing: the whole screen is grey ASCII.
       Touch thumb + index finger together on BOTH hands.
    2. Pull your fingers apart - the 4 fingertips make a shape.
       Everything is ASCII; inside the shape you see the filter.
       The shape follows your fingers. Drop a hand out of view to reset.
    3. Pick a filter by clicking the navbar at the top:
         Photo    : Film, Dreamy Glow, Moody B&W
         Cartoons : Cartoon, Pencil Sketch, Comic Book
         Memes    : Meme Pose Detector (normal camera + a banner naming the meme)
         Animated : Matrix Rain, Snowfall, Glitch, Finger Sparkles
       Or tap (pinch both hands) again to go to the next filter in the category.

Files:
    Filter.py      this file: camera, hand tracking, ASCII, navbar
    common.py      shared colors, font, model downloader
    filters/       one file per filter (+ base.py helpers, __init__.py = navbar list)

Needs hand_landmarker.task and pose_landmarker_lite.task (MediaPipe models)
in the same folder. They are downloaded automatically if missing.

Controls:
    mouse    : click a category / filter in the navbar
    1 - 4    : switch category (Photo, Cartoons, Memes, Animated)
    f        : next filter in the category (same as the pinch tap)
    q or ESC : quit
    + / -    : more / fewer ASCII characters (more = more detail, slower)
    c        : toggle grey <-> bright white ASCII
    i        : invert ASCII brightness
    s        : save a screenshot (ascii_<timestamp>.png)
"""

import time
from types import SimpleNamespace

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import FONT, GREY, model_path
from filters import CATEGORIES      # every filter lives in its own file in filters/

# ----------------------------- CONFIG ---------------------------------------
CAMERA_INDEX = 0          # 0 = default webcam. Try 1, 2 if you have several.
CAMERA_W, CAMERA_H = 1280, 720   # higher res = more accurate hand tracking
START_COLS = 120          # how many characters wide the ASCII image is
CELL_W, CELL_H = 9, 16    # pixel size of one character cell (terminal-like 9x16)
CANDIDATE_CHARS = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
TAP_COOLDOWN = 0.5        # seconds between filter switches (stops double triggers)
WINDOW = "ASCII Cam"

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
    return cv2.merge([cv2.LUT(mono, (np.arange(256) * (c / 255)).astype(np.uint8)) for c in GREY])


def full_ascii(frame, cols, atlas, invert, grey):
    """The whole frame as ASCII, resized back to the camera's size."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ascii_img = colorize(frame_to_ascii(gray, cols, atlas, invert), grey)
    return cv2.resize(ascii_img, (w, h), interpolation=cv2.INTER_AREA)


# ----------------------------- HAND TRACKING --------------------------------
THUMB_TIP, INDEX_TIP = 4, 8      # MediaPipe landmark ids
WRIST, MIDDLE_BASE = 0, 9        # used to measure hand size
PINCH_ON, PINCH_OFF = 0.25, 0.35 # pinch distance / hand size (two thresholds
                                 # so it doesn't flicker at the edge)
LOST_GRACE = 0.6                 # seconds a hand can vanish before the shape resets
# Smoothing (One Euro filter): steady when still, no lag when moving fast
SMOOTH_MIN_CUTOFF = 1.0          # lower = less jitter when hands are still
SMOOTH_BETA = 0.02               # higher = less lag when hands move fast
SHAPE_SCALE = 1.4                # grow the shape out from its center (1.0 = exactly at fingertips)
DETECT_CONFIDENCE = 0.7          # 0..1, higher = fewer false/wobbly detections


def make_hand_detector():
    """Load MediaPipe's hand model (hand_landmarker.task next to this file)."""
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
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


def filter_in_shape(ascii_img, filtered, corners):
    """ASCII everywhere, the filtered image only inside the finger shape."""
    h, w = ascii_img.shape[:2]
    # convexHull keeps the shape from twisting into a bow-tie if fingers cross
    hull = cv2.convexHull(expand_shape(corners).astype(np.int32))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_AA)

    out = cv2.copyTo(filtered, mask, ascii_img.copy())     # filter only where mask is set
    cv2.polylines(out, [hull], True, GREY, 2, cv2.LINE_AA)
    for x, y in corners:
        cv2.circle(out, (int(x), int(y)), 6, (255, 255, 255), -1, cv2.LINE_AA)
    return out


# ----------------------------- NAVBAR ---------------------------------------
NAV_TAB_H, NAV_SUB_H = 40, 34          # category row, filter row (pixels)
NAV_H = NAV_TAB_H + NAV_SUB_H


def nav_buttons(w, cat):
    """Every navbar button as (x0, y0, x1, y1, label, kind, index)."""
    buttons = []
    n = len(CATEGORIES)
    for i, (name, _) in enumerate(CATEGORIES):
        buttons.append((i * w // n, 0, (i + 1) * w // n, NAV_TAB_H, name.upper(), "cat", i))
    filters = CATEGORIES[cat][1]
    m = len(filters)
    for j, f in enumerate(filters):
        buttons.append((j * w // m, NAV_TAB_H, (j + 1) * w // m, NAV_H, f.name, "filter", j))
    return buttons


def draw_navbar(w, ui):
    """Terminal-style bar: selected button = grey block with black text."""
    bar = np.full((NAV_H, w, 3), 18, np.uint8)
    hx, hy = ui["hover"]
    for x0, y0, x1, y1, label, kind, i in nav_buttons(w, ui["cat"]):
        selected = i == (ui["cat"] if kind == "cat" else ui["sel"][ui["cat"]])
        hovered = x0 <= hx < x1 and y0 <= hy < y1
        if selected:
            cv2.rectangle(bar, (x0 + 2, y0 + 3), (x1 - 3, y1 - 4), GREY, -1)
            color = (0, 0, 0)
        elif hovered:
            cv2.rectangle(bar, (x0 + 2, y0 + 3), (x1 - 3, y1 - 4), (55, 55, 55), -1)
            color = (240, 240, 240)
        else:
            color = GREY
        scale = 0.6 if kind == "cat" else 0.5
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        cv2.putText(bar, label, (x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 + th) // 2),
                    FONT, scale, color, 1, cv2.LINE_AA)
    cv2.line(bar, (0, NAV_TAB_H), (w, NAV_TAB_H), (60, 60, 60), 1)
    cv2.line(bar, (0, NAV_H - 1), (w, NAV_H - 1), (90, 90, 90), 1)
    return bar


def on_mouse(event, x, y, flags, ui):
    """Mouse callback: hover highlight + click to pick a category / filter."""
    ui["hover"] = (x, y)
    if event != cv2.EVENT_LBUTTONDOWN or ui["width"] is None:
        return
    for x0, y0, x1, y1, _, kind, i in nav_buttons(ui["width"], ui["cat"]):
        if x0 <= x < x1 and y0 <= y < y1:
            if kind == "cat":
                ui["cat"] = i
            else:
                ui["sel"][ui["cat"]] = i
            return


def next_filter(ui):
    cat = ui["cat"]
    ui["sel"][cat] = (ui["sel"][cat] + 1) % len(CATEGORIES[cat][1])


# ----------------------------- MAIN -----------------------------------------
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

    # navbar state (shared with the mouse callback)
    ui = {"cat": 0, "sel": [0] * len(CATEGORIES), "hover": (-1, -1), "width": None}
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse, ui)

    cols, grey, invert = START_COLS, True, False
    prev, fps = time.time(), 0.0
    start = time.time()
    last_ts = -1
    pinch_state = [False, False]     # per detected hand, for hysteresis
    landed = False                   # True after the double-pinch "landing"
    corners = None                   # smoothed fingertip positions
    smoother = OneEuroFilter()
    last_seen = 0.0                  # last time both hands were visible
    was_tapping = False              # both hands pinched last frame?
    last_tap = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not received, stopping.")
            break

        frame = cv2.flip(frame, 1)                       # mirror, like a selfie
        h, w = frame.shape[:2]
        if ui["width"] is None:                          # first frame: size the window
            ui["width"] = w
            cv2.resizeWindow(WINDOW, w, h + NAV_H)

        # --- detect hands (MediaPipe wants RGB + strictly increasing timestamps) ---
        now = time.time()
        ts = max(int((now - start) * 1000), last_ts + 1)
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hands = sort_hands(detector.detect_for_video(mp_image, ts).hand_landmarks)

        pinched = []
        for i, hand in enumerate(hands):
            r = pinch_ratio(hand)
            was = pinch_state[i] if i < len(pinch_state) else False
            pinched.append(r < (PINCH_OFF if was else PINCH_ON))
        pinch_state = pinched + [False] * (2 - len(pinched))

        if len(hands) == 2:
            last_seen = now
            # A "tap" = both hands pinch, counted once when the pinch starts
            tapping = all(pinched)
            if tapping and not was_tapping and now - last_tap > TAP_COOLDOWN:
                last_tap = now
                if not landed:
                    landed = True                        # landing
                else:
                    next_filter(ui)                      # next filter in category
            was_tapping = tapping
            corners = smoother(fingertips(hands, w, h), now)
        elif now - last_seen > LOST_GRACE:
            # a hand left the camera for too long -> back to waiting
            landed, corners, was_tapping = False, None, False
            smoother = OneEuroFilter()

        current = CATEGORIES[ui["cat"]][1][ui["sel"][ui["cat"]]]
        ctx = SimpleNamespace(t=now - start, dt=min(max(now - prev, 1e-3), 0.1),
                              hands=hands, mp_image=mp_image, timestamp_ms=ts)

        ascii_img = full_ascii(frame, cols, atlas, invert, grey)   # landing screen
        if landed and corners is not None:
            out = filter_in_shape(ascii_img, current.render(frame, ctx), corners)
        else:
            out = ascii_img
            draw_fingers(out, hands, pinched)
        current.overlay(out, ctx)                        # banner / sparkles

        # FPS counter (smoothed)
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        status = ("tap to switch filter" if landed
                  else f"pinch both hands to start | hands {len(hands)}/2")
        cv2.putText(out, f"{fps:4.1f} fps | {CATEGORIES[ui['cat']][0]} > {current.name} | {status}",
                    (8, 22), FONT, 0.5, GREY, 1, cv2.LINE_AA)

        screen = np.vstack([draw_navbar(w, ui), out])
        cv2.imshow(WINDOW, screen)

        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break                                        # window closed with the X
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
        elif ord("1") <= key < ord("1") + len(CATEGORIES):
            ui["cat"] = key - ord("1")
        elif key == ord("f"):
            next_filter(ui)
        elif key == ord("s"):
            name = time.strftime("ascii_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, screen)
            print("Saved", name)

    for _, filters in CATEGORIES:
        for f in filters:
            f.close()
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
