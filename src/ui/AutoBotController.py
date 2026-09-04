# Standard Import
import sys
from argparse import Namespace

# Pyside
from PySide6.QtCore import QObject, Signal

#  Local Import
from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.input.KeyBoardListener import KeyBoardListener
from src.ui.RouteRecorderController import RouteRecorderController
from src.utils.common import load_yaml
from src.utils.logger import logger


class AutoBotController(QObject):
    '''
    AutoBot Controller server as a middleman between engine and UI
    '''
    debug_image_signal = Signal(object)
    route_map_viz_signal = Signal(object)

    def __init__(self):
        """
        Init
        """
        super().__init__()
        self.ui = None

        # Init Auto Bot
        try:
            # Fake args to pass to AutoBot
            args = Namespace(
                disable_control=False,
                cfg="default",
                debug=False,
                record=False,
                is_ui=True,
                disable_viz=True,
                test_image='',
            )
            self.auto_bot = MapleStoryAutoBot(args)
        except Exception as e:
            logger.error(f"MapleStoryAutoBot Init Failed: {e}")
            sys.exit(1)
        else:
            logger.info("MapleStoryAutoBot Init Successfully")

        self.route_recorder_controller = RouteRecorderController(
            can_start=lambda: not (
                self.auto_bot.thread_auto_bot is not None
                and self.auto_bot.thread_auto_bot.is_alive()
            )
        )

        # Update signal for debug window viz
        self.auto_bot.update_signals(self.debug_image_signal,
                                     self.route_map_viz_signal)

        # Monitor function keys
        self.kb_listener = KeyBoardListener(is_autobot=True)

    def update_signal(self, ui):
        '''
        Only called after UI init
        '''
        self.debug_image_signal.connect(ui.update_debug_canvas)
        self.route_map_viz_signal.connect(ui.update_route_map_canvas)
        # Register Function Key handler
        self.kb_listener.register_func_key_handler('f1', ui.handle_f1)
        self.kb_listener.register_func_key_handler('f2', ui.handle_f2)
        self.kb_listener.register_func_key_handler('f3', ui.handle_f3)
        self.kb_listener.register_func_key_handler('f12', lambda: ui.request_close.emit())

    def start_bot(self, cfg_path):
        '''
        Start the bot engine threads
        '''
        if self.route_recorder_controller.is_running:
            logger.error("Stop map creation before starting AutoBot")
            return -1

        # Get config from ui
        cfg = load_yaml(cfg_path)

        # Auto bot load config
        if self.auto_bot.load_config(cfg) != 0:
            return -1 # Load fail

        # Start the bot engine
        try:
            self.auto_bot.start()
        except Exception as e:
            self.auto_bot.terminate_threads()
            logger.error(f"[start_bot] {e}")
            return -1 # Start fail

        return 0 # start bot success

    def pause_bot(self):
        '''
        Gracefully pause in the engine
        '''
        self.auto_bot.pause()

    def take_screenshot(self):
        '''
        Called when user press screenshot button
        '''
        self.auto_bot.screenshot_img_frame()

    def start_recording(self):
        '''
        Called when user press start record button
        '''
        self.auto_bot.start_record()

    def stop_recording(self):
        '''
        Called when user press stop record button
        '''
        self.auto_bot.stop_record()

    def terminate_bot(self):
        '''
        Called when user stop bot or close UI
        '''
        # Terminate all bot threads
        self.auto_bot.terminate_threads()
        self.route_recorder_controller.stop_session()
        self.kb_listener.stop()

    def enable_bot_viz(self):
        '''
        Called when user switch to viz tab
        '''
        self.auto_bot.enable_viz()

    def disable_bot_viz(self):
        '''
        Called when user switch from viz tab
        '''
        self.auto_bot.disable_viz()
