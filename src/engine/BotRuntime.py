"""Lifecycle management for AutoBot worker resources."""

import threading

from src.utils.logger import logger


class BotRuntime:
    def __init__(
        self,
        cfg,
        args,
        disable_control,
        loop_target,
        keyboard_factory,
        capture_factory,
        static_capture_factory,
        health_monitor_factory,
        profiler_factory,
    ):
        self.cfg = cfg
        self.args = args
        self.disable_control = disable_control
        self.loop_target = loop_target
        self.keyboard_factory = keyboard_factory
        self.capture_factory = capture_factory
        self.static_capture_factory = static_capture_factory
        self.health_monitor_factory = health_monitor_factory
        self.profiler_factory = profiler_factory
        self.keyboard = None
        self.capture = None
        self.health_monitor = None
        self.profiler = None
        self.thread = None

    def prepare(self):
        try:
            self.keyboard = self.keyboard_factory(self.cfg)
            if self.disable_control:
                self.keyboard.disable()
            if self.args.test_image:
                self.capture = self.static_capture_factory(
                    self.cfg,
                    self.args.test_image,
                )
            else:
                self.capture = self.capture_factory(self.cfg)
            self.health_monitor = self.health_monitor_factory(
                self.cfg,
                self.keyboard,
            )
            if (
                self.cfg["health_monitor"]["enable"]
                and not self.disable_control
            ):
                self.health_monitor.start()
            self.profiler = self.profiler_factory(self.cfg)
        except Exception:
            self.stop()
            raise

    def start_thread(self):
        self.thread = threading.Thread(
            target=self.loop_target,
            name="MapleStoryAutoBot",
            daemon=True,
        )
        self.thread.start()
        return self.thread

    def stop(self):
        if self.keyboard is not None:
            self.keyboard.is_terminated = True
            self.keyboard.release_all_key()
        if self.health_monitor is not None:
            self.health_monitor.stop()
        if self.capture is not None:
            self.capture.stop()

        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("[MapleStoryAutoBot] Stop timed out")
        if self.keyboard is not None:
            self.keyboard.stop()
