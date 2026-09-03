from pathlib import Path
import threading

from src.utils.common import load_image
from src.utils.logger import logger


class StaticImageCapturor:
    """Capture-compatible source backed by one image from the test directory."""

    def __init__(self, cfg, image_name):
        image_path = Path(image_name)
        if image_path.suffix.lower() != ".png":
            image_path = image_path.with_suffix(".png")
        if image_path.parent == Path("."):
            image_path = Path("test") / image_path

        self.cfg = cfg
        self.window_title = cfg["game_window"]["title"]
        self.lock = threading.Lock()
        self.frame = load_image(str(image_path))
        self.is_terminated = False
        logger.info(f"[StaticImageCapturor] Loaded test image: {image_path}")

    def get_frame(self):
        with self.lock:
            if self.is_terminated:
                return None
            return self.frame.copy()

    def stop(self):
        self.is_terminated = True
        logger.info("[StaticImageCapturor] Terminated")
