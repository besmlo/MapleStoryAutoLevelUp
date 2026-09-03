from argparse import Namespace
import threading

from PySide6.QtCore import QObject, Signal

from src.engine.MapBundleValidator import (
    MapBundleValidator,
    register_map_bundle,
)
from tools.routeRecorder import RouteRecorder


class RouteRecorderController(QObject):
    game_frame_signal = Signal(object)
    map_frame_signal = Signal(object)
    route_frame_signal = Signal(object)
    status_signal = Signal(str)
    error_signal = Signal(str)
    stopped_signal = Signal()

    def __init__(self, can_start=None):
        super().__init__()
        self.can_start = can_start or (lambda: True)
        self.recorder = None
        self.thread = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self._stopped_notified = True

    @property
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def start_session(self, map_id, cfg):
        if self.is_running:
            raise RuntimeError("A map creation session is already running")
        if not self.can_start():
            raise RuntimeError("Pause AutoBot before starting map creation")

        args = Namespace(
            cfg="custom",
            new_map=map_id,
            map="",
            show_debug_windows=False,
            start_recording=False,
            replace_existing=False,
        )
        self.recorder = RouteRecorder(args, cfg=cfg)
        self.stop_event.clear()
        self._stopped_notified = False
        self.thread = threading.Thread(
            target=self._run_loop,
            name="RouteRecorder",
            daemon=True,
        )
        self.thread.start()
        self.status_signal.emit("Scanning minimap. Move around to reveal the map.")

    def set_recording(self, enabled):
        with self.lock:
            self._require_recorder().set_recording(enabled)
        state = "Recording route" if enabled else "Route recording paused"
        self.status_signal.emit(state)

    def finish_route(self):
        with self.lock:
            path = self._require_recorder().finish_route()
        self.status_signal.emit(f"Saved {path}")
        return path

    def save_map(self):
        with self.lock:
            path = self._require_recorder().save_map()
        self.status_signal.emit(f"Saved {path}")
        return path

    def validate_and_register(self, display_name, monsters):
        with self.lock:
            recorder = self._require_recorder()
            validator = MapBundleValidator(recorder.cfg["route"]["color_code"])
            validation = validator.validate(recorder.map_dir)
            if not validation.is_valid:
                raise ValueError("\n".join(validation.errors))
            register_map_bundle(
                recorder.args.new_map,
                display_name,
                monsters,
            )
            map_id = recorder.args.new_map
        self.status_signal.emit(
            f"Map '{map_id}' registered with {validation.route_count} route(s)"
        )
        return map_id

    def stop_session(self):
        self.stop_event.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("Map creation session did not stop in time")
        self._close_recorder()
        self.thread = None
        self._notify_stopped()

    def _run_loop(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    recorder = self._require_recorder()
                    result = recorder.run_once()
                    if result == 0:
                        self._emit_previews(recorder)
                    fps = max(1, recorder.fps_limit)
                self.stop_event.wait(1.0 / fps)
        except Exception as error:
            self.error_signal.emit(str(error))
            self.stop_event.set()
        finally:
            self._close_recorder()
            self.thread = None
            self._notify_stopped()

    def _emit_previews(self, recorder):
        game_frame = None
        map_frame = None
        route_frame = None
        if recorder.img_frame_debug is not None:
            game_frame = recorder.img_frame_debug.copy()
        if recorder.img_map is not None:
            map_frame = recorder.img_map.copy()
        if recorder.img_route_debug is not None:
            route_frame = recorder.img_route_debug.copy()
        self.game_frame_signal.emit(game_frame)
        self.map_frame_signal.emit(map_frame)
        self.route_frame_signal.emit(route_frame)

    def _close_recorder(self):
        with self.lock:
            if self.recorder is not None:
                self.recorder.stop()
                self.recorder = None

    def _require_recorder(self):
        if self.recorder is None:
            raise RuntimeError("Start a map creation session first")
        return self.recorder

    def _notify_stopped(self):
        with self.lock:
            if self._stopped_notified:
                return
            self._stopped_notified = True
        self.stopped_signal.emit()
