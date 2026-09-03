import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from src.ui.ui import MainWindow
from src.ui.RouteRecorderController import RouteRecorderController


class ControllerStub:
    def __init__(self):
        self.visualization_enabled = False
        self.route_recorder_controller = RouteRecorderController()

    def enable_bot_viz(self):
        self.visualization_enabled = True

    def disable_bot_viz(self):
        self.visualization_enabled = False


class MonitoringUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controller = ControllerStub()
        self.window = MainWindow(self.controller)

    def tearDown(self):
        self.window.deleteLater()
        self.app.processEvents()

    def test_visualizations_share_one_monitoring_tab(self):
        tab_names = [
            self.window.tabs.tabText(index)
            for index in range(self.window.tabs.count())
        ]

        self.assertEqual(
            tab_names,
            ["Main", "Advanced Settings", "Monitoring", "Map Creator"],
        )
        self.assertIsNotNone(self.window.debug_canvas)
        self.assertIsNotNone(self.window.route_map_canvas)

    def test_monitoring_tab_controls_visualization_updates(self):
        monitoring_index = self.window.tabs.indexOf(self.window.tab_monitoring)
        self.window.tabs.setCurrentIndex(monitoring_index)
        self.assertTrue(self.controller.visualization_enabled)

        self.window.tabs.setCurrentIndex(0)
        self.assertFalse(self.controller.visualization_enabled)

    def test_canvases_accept_frames_and_empty_route_preview(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)

        self.window.update_debug_canvas(frame)
        self.window.update_route_map_canvas(frame)
        self.assertFalse(self.window.debug_canvas.pixmap().isNull())
        self.assertFalse(self.window.route_map_canvas.pixmap().isNull())

        self.window.update_route_map_canvas(None)
        self.assertEqual(
            self.window.route_map_canvas.text(),
            "Route preview is available in normal mode",
        )


if __name__ == "__main__":
    unittest.main()
