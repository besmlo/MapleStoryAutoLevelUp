from argparse import Namespace
import time
import unittest
from unittest.mock import patch

import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.input.StaticImageCapturor import StaticImageCapturor
from src.utils.common import load_yaml, normalize_language_code


class FakeKeyboardController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_terminated = False
        self.is_enable = True
        self.stop_called = False

    def disable(self):
        self.is_enable = False

    def release_all_key(self):
        pass

    def stop(self):
        self.stop_called = True
        self.is_terminated = True


class FakeCapture:
    def __init__(self, cfg, *args):
        self.cfg = cfg
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeHealthMonitor:
    def __init__(self, cfg, keyboard):
        self.cfg = cfg
        self.keyboard = keyboard
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True


class FakeProfiler:
    def __init__(self, cfg):
        self.cfg = cfg


class StabilityTest(unittest.TestCase):
    def test_global_language_uses_existing_asset_suffix(self):
        cfg = load_yaml("config/config_global.yaml")
        self.assertEqual(cfg["system"]["language"], "eng")
        self.assertEqual(normalize_language_code("english"), "eng")
        self.assertEqual(normalize_language_code("zh"), "cn")

    def test_static_image_capture_returns_independent_frames(self):
        source = np.zeros((10, 12, 3), dtype=np.uint8)
        cfg = {"game_window": {"title": "MapleStory Worlds"}}

        with patch(
            "src.input.StaticImageCapturor.load_image", return_value=source
        ):
            capture = StaticImageCapturor(cfg, "sample")

        first = capture.get_frame()
        first[0, 0] = 255
        self.assertTrue(np.all(capture.get_frame()[0, 0] == 0))

        capture.stop()
        self.assertIsNone(capture.get_frame())

    def test_bot_can_stop_cleanly_and_start_again(self):
        args = Namespace(
            disable_control=True,
            disable_viz=True,
            is_ui=True,
            test_image="",
        )
        bot = MapleStoryAutoBot(args)
        bot.cfg = {
            "health_monitor": {"enable": False},
            "game_window": {"title": "MapleStory Worlds"},
        }

        def controlled_loop():
            while not bot.kb.is_terminated:
                time.sleep(0.001)

        bot.loop = controlled_loop
        replacements = {
            "KeyBoardController": FakeKeyboardController,
            "GameWindowCapturor": FakeCapture,
            "HealthMonitor": FakeHealthMonitor,
            "Profiler": FakeProfiler,
        }

        with patch.multiple(
            "src.engine.MapleStoryAutoLevelUp", **replacements
        ):
            bot.start()
            first_keyboard = bot.kb
            self.assertTrue(bot.thread_auto_bot.is_alive())
            with self.assertRaisesRegex(RuntimeError, "already running"):
                bot.start()

            bot.pause()
            self.assertFalse(bot.thread_auto_bot.is_alive())
            self.assertTrue(first_keyboard.stop_called)

            bot.start()
            self.assertIsNot(bot.kb, first_keyboard)
            bot.pause()
            self.assertFalse(bot.thread_auto_bot.is_alive())


if __name__ == "__main__":
    unittest.main()
