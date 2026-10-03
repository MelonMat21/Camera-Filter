"""
All filters, grouped the way the navbar shows them.

Each filter lives in its own file in this folder. A filter's file is only
imported (and the filter created) the first time it's actually used, so
startup is quicker and unused filters never take up memory.

To add a filter: make a new file with a class that extends base.Filter,
then add one line for it below.
"""

import importlib


class LazyFilter:
    """Stands in for a filter until it's first used, then loads it."""

    def __init__(self, name, module, class_name):
        self.name = name                     # shown in the navbar (no loading needed)
        self._module, self._class_name = module, class_name
        self._filter = None

    def load(self):
        if self._filter is None:
            module = importlib.import_module(f".{self._module}", __name__)
            self._filter = getattr(module, self._class_name)()
        return self._filter

    def render(self, frame, ctx):
        return self.load().render(frame, ctx)

    def overlay(self, out, ctx):
        self.load().overlay(out, ctx)

    def close(self):
        if self._filter is not None and hasattr(self._filter, "close"):
            self._filter.close()


# (category, [filters]) - this is what the navbar shows
CATEGORIES = [
    ("Photo", [
        LazyFilter("Film", "film", "Film"),
        LazyFilter("Dreamy Glow", "dreamy_glow", "DreamyGlow"),
        LazyFilter("Moody B&W", "moody_bw", "MoodyBW"),
    ]),
    ("Cartoons", [
        LazyFilter("Cartoon", "cartoon", "Cartoon"),
        LazyFilter("Pencil Sketch", "pencil_sketch", "PencilSketch"),
        LazyFilter("Comic Book", "comic_book", "ComicBook"),
    ]),
    ("Memes", [
        LazyFilter("Meme Pose Detector", "meme_pose_detector", "MemePoseDetector"),
    ]),
    ("Animated", [
        LazyFilter("Matrix Rain", "matrix_rain", "MatrixRain"),
        LazyFilter("Snowfall", "snowfall", "Snowfall"),
        LazyFilter("Glitch", "glitch", "Glitch"),
        LazyFilter("Finger Sparkles", "finger_sparkles", "FingerSparkles"),
    ]),
]
