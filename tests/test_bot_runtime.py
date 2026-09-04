import unittest
from argparse import Namespace

from src.engine.BotRuntime import BotRuntime


class FakeKeyboard:
    def __init__(self, cfg):
        self.disabled = False
        self.is_terminated = False
        self.released = False
        self.stopped = False

    def disable(self):
        self.disabled = True

    def release_all_key(self):
        self.released = True

    def stop(self):
        self.stopped = True


class BotRuntimeTest(unittest.TestCase):
    def make_runtime(self, capture_factory, disable_control=True):
        return BotRuntime(
            cfg={"health_monitor": {"enable": False}},
            args=Namespace(test_image=""),
            disable_control=disable_control,
            loop_target=lambda: None,
            keyboard_factory=FakeKeyboard,
            capture_factory=capture_factory,
            static_capture_factory=lambda cfg, path: object(),
            health_monitor_factory=lambda cfg, keyboard: object(),
            profiler_factory=lambda cfg: object(),
        )

    def test_prepare_disables_control_for_safe_debugging(self):
        runtime = self.make_runtime(lambda cfg: object())
        runtime.prepare()
        self.assertTrue(runtime.keyboard.disabled)

    def test_prepare_failure_releases_and_stops_keyboard(self):
        def fail_capture(cfg):
            raise RuntimeError("capture failed")

        runtime = self.make_runtime(fail_capture)
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            runtime.prepare()

        self.assertTrue(runtime.keyboard.is_terminated)
        self.assertTrue(runtime.keyboard.released)
        self.assertTrue(runtime.keyboard.stopped)


if __name__ == "__main__":
    unittest.main()
