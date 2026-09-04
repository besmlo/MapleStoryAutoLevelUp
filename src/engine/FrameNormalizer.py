"""Normalize captured game frames to the engine's canonical resolution."""

from dataclasses import dataclass

import cv2

from src.utils.common import is_img_16_to_9

CANONICAL_SIZE = (1296, 759)


@dataclass(frozen=True)
class NormalizedFrame:
    image: object = None
    error: str | None = None
    was_resized: bool = False


def normalize_game_frame(frame, cfg, skip_aspect_check=False):
    if frame is None:
        return NormalizedFrame(error="Failed to capture game frame.")
    if not skip_aspect_check and not is_img_16_to_9(frame, cfg):
        return NormalizedFrame(
            error=(
                f"Unexpected window aspect ratio: {frame.shape[:2]}. "
                "Please use a 16:9 windowed resolution."
            )
        )

    was_resized = cfg["game_window"]["size"] != frame.shape[:2]
    image = cv2.resize(
        frame,
        CANONICAL_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )
    return NormalizedFrame(image=image, was_resized=was_resized)
