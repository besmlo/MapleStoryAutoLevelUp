import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from src.ui.MapCreationWidget import MapCreationWidget


class FakeRouteRecorderController(QObject):
    game_frame_signal = Signal(object)
    map_frame_signal = Signal(object)
    route_frame_signal = Signal(object)
    status_signal = Signal(str)
    error_signal = Signal(str)
    stopped_signal = Signal()

    def __init__(self):
        super().__init__()
        self.started_with = None
        self.recording = False
        self.saved_map = False
        self.finished_routes = 0
        self.stopped = False

    def start_session(self, map_id, cfg):
        self.started_with = (map_id, cfg)

    def set_recording(self, enabled):
        self.recording = enabled

    def finish_route(self):
        self.finished_routes += 1

    def save_map(self):
        self.saved_map = True

    def validate_and_register(self, display_name, monsters):
        return self.started_with[0]

    def stop_session(self):
        self.stopped = True
        self.stopped_signal.emit()


class MapCreationWidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controller = FakeRouteRecorderController()
        self.widget = MapCreationWidget(
            self.controller,
            lambda: {"route": {"color_code": {}}},
            {},
        )

    def tearDown(self):
        self.widget.deleteLater()
        self.app.processEvents()

    def test_session_controls_follow_start_and_stop(self):
        self.assertTrue(self.widget.start_button.isEnabled())
        self.assertFalse(self.widget.save_map_button.isEnabled())

        self.widget.map_id_input.setText("test_map")
        self.widget.start_button.click()

        self.assertEqual(self.controller.started_with[0], "test_map")
        self.assertFalse(self.widget.start_button.isEnabled())
        self.assertTrue(self.widget.save_map_button.isEnabled())

        self.widget.record_button.click()
        self.assertTrue(self.controller.recording)
        self.widget.finish_route_button.click()
        self.assertFalse(self.controller.recording)
        self.assertEqual(self.controller.finished_routes, 1)

        self.widget.stop_button.click()
        self.assertTrue(self.controller.stopped)
        self.assertTrue(self.widget.start_button.isEnabled())
        self.assertFalse(self.widget.save_map_button.isEnabled())

    def test_invalid_map_id_does_not_start(self):
        self.widget.map_id_input.setText("../outside")
        self.widget.start_button.click()

        self.assertIsNone(self.controller.started_with)
        self.assertFalse(self.widget.error_label.isHidden())


if __name__ == "__main__":
    unittest.main()
