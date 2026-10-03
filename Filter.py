"""
ascii_cam.py - Real-time webcam -> ASCII filter (grey on black, like a terminal)

Install:   pip install opencv-python numpy
Run:       python ascii_cam.py

Controls:
    q or ESC : quit
    + / -    : more / fewer characters (more = more detail, slower)
    c        : toggle grey <-> bright white
    i        : invert brightness (dense characters for dark areas)
    s        : save a screenshot (ascii_<timestamp>.png)
"""

import time
import cv2
import numpy as np

# ----------------------------- CONFIG ---------------------------------------
CAMERA_INDEX = 0          # 0 = default webcam. Try 1, 2 if you have several.
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


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}. "
                         "Check permissions or try another CAMERA_INDEX.")

    atlas = build_glyph_atlas(CANDIDATE_CHARS, CELL_W, CELL_H)
    print(f"Using {len(atlas)} distinct brightness levels.")

    cols, grey, invert = START_COLS, True, False
    prev, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not received, stopping.")
            break

        frame = cv2.flip(frame, 1)                       # mirror, like a selfie
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mono = frame_to_ascii(gray, cols, atlas, invert)
        out = colorize(mono, grey)

        # FPS counter (smoothed)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        cv2.putText(out, f"{fps:4.1f} fps | {cols} cols", (8, 18),
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()