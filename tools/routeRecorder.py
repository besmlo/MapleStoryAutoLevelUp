"""Standalone command-line entry point for the route recorder."""

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.RouteRecorderEngine import RouteRecorderEngine
from src.utils.logger import logger


class RouteRecorder(RouteRecorderEngine):
    """Backward-compatible name for external scripts and imports."""


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new_map",
        type=str,
        default="new_map",
        help="Specify the new map name",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="custom",
        help="Choose customized config yaml file in config/",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="",
        help="Use this map instead of creating a new one",
    )
    return parser


def main():
    try:
        recorder = RouteRecorder(build_argument_parser().parse_args())
    except Exception as error:
        logger.error(f"RouteRecorder init failed: {error}")
        return 1

    try:
        while True:
            started_at = time.time()
            recorder.run_once()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_duration = time.time() - started_at
            target_duration = 1.0 / recorder.fps_limit
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)
    finally:
        recorder.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
