import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.engine.RouteRecorderEngine import RouteRecorderEngine
from tools.routeRecorder import RouteRecorder


class RouteRecorderCoreTest(unittest.TestCase):
    def test_finish_route_adds_goal_and_resets_route(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RouteRecorderEngine.__new__(RouteRecorderEngine)
            recorder.color_code = {
                (255, 0, 0): {"command": "none none goal"}
            }
            recorder.img_map = np.zeros((20, 30, 3), dtype=np.uint8)
            recorder.img_route = recorder.img_map.copy()
            recorder.img_route[5, 5] = (0, 255, 0)
            recorder.loc_player_global = (12, 10)
            recorder.loc_player_global_last = (5, 5)
            recorder.map_dir = directory
            recorder.idx_routes = 0

            route_path = recorder.finish_route()
            saved_route = cv2.imread(route_path, cv2.IMREAD_COLOR)

            self.assertEqual(Path(route_path).name, "route1.png")
            self.assertTrue(np.array_equal(saved_route[10, 12], (0, 0, 255)))
            self.assertEqual(recorder.idx_routes, 1)
            self.assertIsNone(recorder.loc_player_global_last)
            self.assertTrue(np.array_equal(recorder.img_route, recorder.img_map))

    def test_save_map_requires_scan_result(self):
        recorder = RouteRecorderEngine.__new__(RouteRecorderEngine)
        recorder.img_map = None

        with self.assertRaisesRegex(RuntimeError, "Scan the minimap"):
            recorder.save_map()

    def test_constructor_releases_keyboard_if_capture_fails(self):
        class FakeKeyboard:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        keyboard = FakeKeyboard()
        args = Namespace(
            new_map="capture_failure",
            map="",
            show_debug_windows=False,
            start_recording=False,
            replace_existing=False,
        )
        cfg = {
            "route": {"color_code": {}, "color_code_up_down": {}},
            "route_recoder": {"map_padding": 10},
            "system": {"fps_limit_route_recorder": 30},
        }

        with patch(
                "src.engine.RouteRecorderEngine.os.path.exists",
                return_value=False,
        ), patch("src.engine.RouteRecorderEngine.os.makedirs"), \
                patch(
                    "src.engine.RouteRecorderEngine.KeyBoardListener",
                    return_value=keyboard,
                ), \
                patch(
                    "src.engine.RouteRecorderEngine.GameWindowCapturor",
                    side_effect=RuntimeError("capture failed"),
                ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                RouteRecorder(args, cfg=cfg)

        self.assertTrue(keyboard.stopped)

    def test_command_line_name_remains_compatible(self):
        self.assertTrue(issubclass(RouteRecorder, RouteRecorderEngine))


if __name__ == "__main__":
    unittest.main()
