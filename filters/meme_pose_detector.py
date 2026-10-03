"""Memes > Meme Pose Detector: normal camera + a banner naming the meme you act out."""

from collections import Counter, deque

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import FONT, GREY, model_path
from .base import Filter

MEME_VOTES = 8           # look at the last N frames...
MEME_VOTES_NEEDED = 5    # ...and need this many to agree before showing a meme
MEME_HOLD = 1.0          # keep the banner this many seconds after you stop
VISIBLE = 0.5            # pose landmark visibility needed to trust it
MEME_LIST = ["T-Pose", "Dab", "Facepalm", "Roll Safe", "Shush",
             "Thinking", "Mind Blown", "Praise the Sun", "Arms Crossed"]


def make_pose_detector():
    """Load MediaPipe's body pose model (used by the Meme Pose Detector)."""
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path("pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _finger_out(hand, tip, pip):
    """True if a finger is stretched out (tip clearly farther from wrist than its middle joint)."""
    return bool(np.linalg.norm(hand[tip] - hand[0]) > np.linalg.norm(hand[pip] - hand[0]) * 1.1)


def detect_meme(pose, hands, w, h):
    """
    Name the meme pose being acted out, or None.
    pose  : 33 MediaPipe pose landmarks (or None)
    hands : list of 21-landmark hands
    Every distance is measured in shoulder widths (S), so it works near or far.
    """
    if pose is None:
        return None

    def P(i):
        return np.array([pose[i].x * w, pose[i].y * h])

    def seen(*ids):
        return all((pose[i].visibility or 0) > VISIBLE for i in ids)

    if not seen(0, 11, 12):                     # need nose + both shoulders
        return None
    ls, rs = P(11), P(12)
    S = np.linalg.norm(ls - rs)
    if S < 10:
        return None
    nose = P(0)
    mid = (ls + rs) / 2                         # middle of the shoulders
    eyes = (P(2) + P(5)) / 2
    mouth = (P(9) + P(10)) / 2
    face = (eyes + nose) / 2

    # arms: (shoulder, elbow, wrist, elbow seen?, wrist seen?)
    arms = [(P(s), P(e), P(wr), seen(e), seen(wr)) for s, e, wr in ((11, 13, 15), (12, 14, 16))]

    def own_side(p, shoulder):                  # p is on the same side as that shoulder
        return (p[0] - mid[0]) * (shoulder[0] - mid[0]) > 0

    hand_pts = [np.array([[lm.x * w, lm.y * h] for lm in hand]) for hand in hands]

    def palm(hp):
        return hp[[0, 5, 9, 13, 17]].mean(axis=0)

    # 1. Praise the Sun: both arms straight up in a wide V
    if all(a[4] for a in arms):
        w1, w2 = arms[0][2], arms[1][2]
        if w1[1] < nose[1] - 0.6 * S and w2[1] < nose[1] - 0.6 * S and abs(w1[0] - w2[0]) > 1.8 * S:
            return "Praise the Sun"

    # 2. Mind Blown: both hands at the sides/top of the head
    if len(hand_pts) == 2:
        if all(np.linalg.norm(palm(hp) - nose) < 1.3 * S and palm(hp)[1] < eyes[1] + 0.2 * S
               and abs(palm(hp)[0] - nose[0]) > 0.45 * S for hp in hand_pts):
            return "Mind Blown"

    # 3. Dab: one arm stretched up and out, the other elbow up at the face
    for (sa, ea, wa, ea_ok, wa_ok), (sb, eb, wb, eb_ok, wb_ok) in (arms, arms[::-1]):
        if wa_ok and eb_ok:
            up_out = (wa[1] < sa[1] - 0.3 * S and np.linalg.norm(wa - sa) > 1.2 * S
                      and own_side(wa, sa) and abs(wa[0] - mid[0]) > 1.0 * S)
            face_in_elbow = np.linalg.norm(eb - nose) < 0.9 * S and eb[1] < mid[1]
            if up_out and face_in_elbow:
                return "Dab"

    # 4. T-Pose: both arms straight out to the sides at shoulder height
    def arm_out(s, e, wr, e_ok, wr_ok):
        if wr_ok:
            return (abs(wr[1] - s[1]) < 0.5 * S and own_side(wr, s)
                    and abs(wr[0] - mid[0]) > 1.4 * S and (not e_ok or abs(e[1] - s[1]) < 0.5 * S))
        if e_ok:   # wrist out of frame: the elbow must be straight out instead
            return abs(e[1] - s[1]) < 0.4 * S and own_side(e, s) and abs(e[0] - mid[0]) > 1.0 * S
        return False
    if all(arm_out(*a) for a in arms):
        return "T-Pose"

    for hp in hand_pts:
        index_out = _finger_out(hp, 8, 6)
        middle_out = _finger_out(hp, 12, 10)
        ring_out = _finger_out(hp, 16, 14)
        pinky_out = _finger_out(hp, 20, 18)
        tip = hp[8]

        # 5. Shush: one finger up on the lips
        if index_out and not middle_out and not ring_out and np.linalg.norm(tip - mouth) < 0.35 * S:
            return "Shush"
        # 6. Facepalm: open hand covering the face
        if sum((index_out, middle_out, ring_out, pinky_out)) >= 3 and np.linalg.norm(palm(hp) - face) < 0.5 * S:
            return "Facepalm"
        # 7. Roll Safe: finger tapping the side of the head
        if (index_out and not middle_out and eyes[1] - 0.6 * S < tip[1] < eyes[1] + 0.25 * S
                and 0.3 * S < abs(tip[0] - nose[0]) < 0.9 * S):
            return "Roll Safe"
        # 8. Thinking: hand under the chin
        chin = mouth + np.array([0, 0.18 * S])
        top = hp[hp[:, 1].argmin()]                 # highest point of the hand
        if (np.linalg.norm(top - chin) < 0.3 * S and palm(hp)[1] > mouth[1]
                and abs(palm(hp)[0] - nose[0]) < 0.6 * S):
            return "Thinking"

    # 9. Arms Crossed: each wrist across the body, at chest height
    if all(a[3] and a[4] for a in arms):
        if all(not own_side(wr, s) and mid[1] < wr[1] < mid[1] + 1.8 * S and e[1] > mid[1]
               for s, e, wr, _, _ in arms):
            return "Arms Crossed"
    return None


def draw_banner(out, title, subtitle):
    """Dark bar at the bottom of the screen with big + small centered text."""
    h, w = out.shape[:2]
    bh = 86
    region = out[h - bh:]
    region[:] = (region * 0.3).astype(np.uint8)
    cv2.line(out, (0, h - bh), (w, h - bh), GREY, 1)
    for text, y, scale, thick, color in ((title, h - bh + 42, 1.3, 3, (255, 255, 255)),
                                         (subtitle, h - 16, 0.55, 1, GREY)):
        while scale > 0.3 and cv2.getTextSize(text, FONT, scale, thick)[0][0] > w - 20:
            scale -= 0.05                       # shrink long text to fit
        tw = cv2.getTextSize(text, FONT, scale, thick)[0][0]
        cv2.putText(out, text, ((w - tw) // 2, y), FONT, scale, color, thick, cv2.LINE_AA)


class MemePoseDetector(Filter):
    """Normal camera, plus a banner naming the meme pose you're acting out."""
    name = "Meme Pose Detector"

    def __init__(self):
        self.pose = None                       # loaded the first time it's used
        self.votes = deque(maxlen=MEME_VOTES)
        self.current, self.last_seen = None, 0.0

    def overlay(self, out, ctx):
        if self.pose is None:
            self.pose = make_pose_detector()
        result = self.pose.detect_for_video(ctx.mp_image, ctx.timestamp_ms)
        pose = result.pose_landmarks[0] if result.pose_landmarks else None
        h, w = out.shape[:2]

        self.votes.append(detect_meme(pose, ctx.hands, w, h))
        counts = Counter(v for v in self.votes if v).most_common(1)
        if counts and counts[0][1] >= MEME_VOTES_NEEDED:
            self.current, self.last_seen = counts[0][0], ctx.t
        elif ctx.t - self.last_seen > MEME_HOLD:
            self.current = None

        if self.current:
            draw_banner(out, self.current.upper(), "meme detected")
        elif pose is None:
            draw_banner(out, "NO BODY FOUND", "step back so your head and shoulders are in view")
        else:
            draw_banner(out, "ACT OUT A MEME", "try: " + ", ".join(MEME_LIST))

    def close(self):
        if self.pose is not None:
            self.pose.close()
