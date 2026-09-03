import copy
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.engine.MapBundleValidator import validate_map_id
from src.utils.ui import clear_debug_canvas, update_image_canvas


class MapCreationWidget(QWidget):
    map_created = Signal(str)

    def __init__(self, controller, config_provider, translations):
        super().__init__()
        self.controller = controller
        self.config_provider = config_provider
        self.translations = translations
        self._build_ui()
        self._connect_controller()
        self._set_session_active(False)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        setup_group = QGroupBox("1. Map setup")
        setup_layout = QFormLayout(setup_group)
        self.map_id_input = QLineEdit()
        self.map_id_input.setPlaceholderText("Example: north_forest_training_ground_2")
        self.display_name_input = QLineEdit()
        self.display_name_input.setPlaceholderText("Optional display name")
        self.monster_list = QListWidget()
        self.monster_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        for monster_id in self._available_monsters():
            translated = self.translations.get(monster_id, monster_id)
            self.monster_list.addItem(f"{monster_id} ({translated})")
        self.start_button = QPushButton("Start map scan")
        setup_layout.addRow("Map ID", self.map_id_input)
        setup_layout.addRow("Display name", self.display_name_input)
        setup_layout.addRow("Monsters", self.monster_list)
        setup_layout.addRow(self.start_button)
        layout.addWidget(setup_group)

        control_layout = QHBoxLayout()
        self.save_map_button = QPushButton("Save map")
        self.record_button = QPushButton("Start route recording")
        self.record_button.setCheckable(True)
        self.finish_route_button = QPushButton("Finish current route")
        self.complete_button = QPushButton("Validate and finish")
        self.stop_button = QPushButton("Stop session")
        for button in (
            self.save_map_button,
            self.record_button,
            self.finish_route_button,
            self.complete_button,
            self.stop_button,
        ):
            control_layout.addWidget(button)
        layout.addLayout(control_layout)

        self.status_label = QLabel("Enter a new Map ID to begin.")
        self.status_label.setWordWrap(True)
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.status_label)
        layout.addWidget(self.error_label)

        previews = QSplitter()
        game_panel, self.game_canvas = self._create_preview("Game detection")
        map_panel, self.map_canvas = self._create_preview("Scanned map")
        route_panel, self.route_canvas = self._create_preview("Current route")
        previews.addWidget(game_panel)
        previews.addWidget(map_panel)
        previews.addWidget(route_panel)
        previews.setStretchFactor(0, 2)
        previews.setStretchFactor(1, 1)
        previews.setStretchFactor(2, 1)
        layout.addWidget(previews, 1)

        self.start_button.clicked.connect(self._start_session)
        self.save_map_button.clicked.connect(self._save_map)
        self.record_button.toggled.connect(self._toggle_recording)
        self.finish_route_button.clicked.connect(self._finish_route)
        self.complete_button.clicked.connect(self._complete)
        self.stop_button.clicked.connect(self._stop_session)

    def _create_preview(self, title):
        panel = QGroupBox(title)
        layout = QVBoxLayout(panel)
        canvas = QLabel()
        canvas.setMinimumSize(180, 140)
        canvas.setStyleSheet("background-color: black; color: white;")
        clear_debug_canvas(canvas, "Waiting for map scan")
        layout.addWidget(canvas)
        return panel, canvas

    def _available_monsters(self):
        if not os.path.isdir("monster"):
            return []
        return sorted(
            name
            for name in os.listdir("monster")
            if os.path.isdir(os.path.join("monster", name))
        )

    def _connect_controller(self):
        self.controller.game_frame_signal.connect(
            lambda image: self._update_canvas(self.game_canvas, image)
        )
        self.controller.map_frame_signal.connect(
            lambda image: self._update_canvas(self.map_canvas, image)
        )
        self.controller.route_frame_signal.connect(
            lambda image: self._update_canvas(self.route_canvas, image)
        )
        self.controller.status_signal.connect(self.status_label.setText)
        self.controller.error_signal.connect(self._show_error)
        self.controller.stopped_signal.connect(
            lambda: self._set_session_active(False)
        )

    def _start_session(self):
        try:
            map_id = validate_map_id(self.map_id_input.text().strip())
            self.error_label.setVisible(False)
            self.controller.start_session(
                map_id, copy.deepcopy(self.config_provider())
            )
        except Exception as error:
            self._show_error(str(error))
            return
        self._set_session_active(True)

    def _save_map(self):
        self._run_action(self.controller.save_map)

    def _toggle_recording(self, enabled):
        self.record_button.setText(
            "Pause route recording" if enabled else "Start route recording"
        )
        self._run_action(lambda: self.controller.set_recording(enabled))

    def _finish_route(self):
        if self.record_button.isChecked():
            self.record_button.setChecked(False)
        self._run_action(self.controller.finish_route)

    def _complete(self):
        monsters = [
            item.text().split(" (")[0]
            for item in self.monster_list.selectedItems()
        ]
        try:
            self.controller.save_map()
            map_id = self.controller.validate_and_register(
                self.display_name_input.text().strip(), monsters
            )
        except Exception as error:
            self._show_error(str(error))
            return
        self.controller.stop_session()
        self.map_created.emit(map_id)
        self.status_label.setText(f"Map '{map_id}' is ready to use.")

    def _stop_session(self):
        self.controller.stop_session()
        self.status_label.setText("Map creation session stopped.")

    def _run_action(self, action):
        try:
            self.error_label.setVisible(False)
            action()
        except Exception as error:
            self._show_error(str(error))

    def _show_error(self, message):
        self.error_label.setText(message)
        self.error_label.setVisible(True)

    def _set_session_active(self, active):
        self.map_id_input.setEnabled(not active)
        self.display_name_input.setEnabled(not active)
        self.monster_list.setEnabled(not active)
        self.start_button.setEnabled(not active)
        for button in (
            self.save_map_button,
            self.record_button,
            self.finish_route_button,
            self.complete_button,
            self.stop_button,
        ):
            button.setEnabled(active)
        if not active:
            self.record_button.blockSignals(True)
            self.record_button.setChecked(False)
            self.record_button.setText("Start route recording")
            self.record_button.blockSignals(False)

    @staticmethod
    def _update_canvas(canvas, image):
        if image is not None:
            update_image_canvas(canvas, image)
