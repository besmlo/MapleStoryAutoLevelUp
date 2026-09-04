'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import argparse
import datetime
import logging
import os
import random
import sys
import threading
import time

import cv2

# Library import
import numpy as np
import yaml

from src.input.KeyBoardController import KeyBoardController, press_key
from src.input.KeyBoardListener import KeyBoardListener
from src.input.StaticImageCapturor import StaticImageCapturor
from src.utils.common import (
    activate_game_window,
    click_in_game_window,
    draw_rectangle,
    find_pattern_sqdiff,
    get_all_other_player_locations_on_minimap,
    get_minimap_loc_size,
    get_player_location_on_minimap,
    is_mac,
    load_yaml,
    override_cfg,
    screenshot,
)

# Local import
from src.utils.logger import logger

if is_mac():
    from src.input.GameWindowCapturorForMac import GameWindowCapturor
else:
    from src.input.GameWindowCapturor import GameWindowCapturor
from src.engine.BotConfigLoader import ConfigResourceError, load_bot_resources
from src.engine.BotRuntime import BotRuntime
from src.engine.CombatAnalyzer import CombatAnalyzer
from src.engine.FiniteStateMachine import FiniteStateMachine
from src.engine.FrameNormalizer import normalize_game_frame
from src.engine.HealthMonitor import HealthMonitor
from src.engine.MonsterDetector import MonsterDetector
from src.engine.PlayerLocator import PlayerLocator
from src.engine.Profiler import Profiler
from src.engine.RouteAnalyzer import find_nearest_actions
from src.states.auxiliary import AuxiliaryState
from src.states.hunting import HuntingState
from src.states.patrol import PatrolState

BOT_MODE_STATES = {
    "normal": "hunting",
    "aux": "aux",
    "patrol": "patrol",
}

class MapleStoryAutoBot:
    '''
    MapleStoryAutoBot
    '''
    def __init__(self, args):
        '''
        Init MapleStoryAutoBot
        '''
        self.args = args # User args
        self.cfg = None # Configuration
        self.idx_routes = 0 # Index of route map
        self.monsters_info = {} # monster information
        self.monsters = [] # monster detected in current frame
        self.fps = 0 # Frame per second
        self.red_dot_center_prev = None # previous other player location in minimap
        self.video_writer = None # For video recording feature
        self.color_code = {} # For color code instruction
        self.color_code_up_down = {} # Color code only contain 'up' and 'down'
        self.thread_auto_bot = None # thread for running autobot
        self.lifecycle_lock = threading.RLock()
        self.cmd_move_x = "none" # "left" "right"
        self.cmd_move_y = "none" # "up" "down"
        self.cmd_action = "none" # "jump" "attack" ....
        # Signals (for UI)
        self.image_debug_signal = None
        self.route_map_viz_signal = None
        # Flags
        self.is_first_frame = True # first frame flag
        self.is_terminated = False # Close all object and thread if True
        self.is_on_ladder = False # Character is on ladder or not
        self.is_show_debug_window = not args.disable_viz #
        self.is_need_show_debug_window = not args.disable_viz #
        self.is_disable_control = args.disable_control or bool(args.test_image)
        self.is_ui = args.is_ui # Whether is using UI framework to invoke engine
        self.is_frame_done = False #
        # Coordinate (top-left coordinate)
        self.loc_nametag = (0, 0) # nametag location on game screen
        self.loc_party_red_bar = (0, 0) # party red bar location on game screen
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_watch_dog = (0, 0) # watch dog location on global map
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer
        self.t_last_attack = time.time() # Last attack timer for cooldown
        self.t_last_minimap_update = time.time()
        self.t_to_change_channel = time.time()
        # Images
        self.img_map = None
        self.img_routes = []
        self.img_nametag = None
        self.img_nametag_gray = None
        self.img_create_party_enable = None
        self.img_create_party_disable = None
        self.img_login_button = None

        # Database
        self.data = load_yaml("config/config_data.yaml")
        # Threads & Objects
        self.kb = None # Keyboard controller
        self.capture = None # Game window capturor
        self.health_monitor = None # Health monitor
        self.profiler = None # Profiler, for performance issue debugging
        self.runtime = None

        # Finite State Machine
        self.fsm = FiniteStateMachine()
        self.fsm.add_state(HuntingState    ("hunting"     , self))
        self.fsm.add_state(AuxiliaryState  ("aux"         , self))
        self.fsm.add_state(PatrolState     ("patrol"      , self))
        self.fsm.set_state("hunting")

    def update_signals(self, image_debug_signal, route_map_viz_signal):
        '''
        Update signal from UI framework.
        For debug window viz
        '''
        self.image_debug_signal = image_debug_signal
        self.route_map_viz_signal = route_map_viz_signal

    def load_config(self, cfg):
        '''
        load_config
        '''
        mode = cfg["bot"]["mode"]
        if mode not in BOT_MODE_STATES:
            logger.error(f"Unsupported bot mode: {mode}")
            return -1

        try:
            resources = load_bot_resources(cfg, self.data)
        except ConfigResourceError as error:
            logger.error(str(error))
            return -1

        self.color_code = resources.color_code
        self.color_code_up_down = resources.color_code_up_down
        self.img_map = resources.img_map
        self.img_routes = resources.img_routes
        self.monsters_info = resources.monsters_info
        self.img_nametag = resources.img_nametag
        self.img_nametag_gray = resources.img_nametag_gray
        self.img_create_party_enable = resources.img_create_party_enable
        self.img_create_party_disable = resources.img_create_party_disable
        self.img_login_button = resources.img_login_button
        if self.monsters_info:
            logger.info(f"Loaded monsters: {list(self.monsters_info.keys())}")

        # Print mode on log
        logger.info(f"[load_config] Config AutoBot as {cfg['bot']['mode']} mode")

        # Update cfg
        self.cfg = cfg
        self.fsm.set_state(BOT_MODE_STATES[mode])

        return 0 # load successfully

    def start(self):
        '''
        Start all threads
        '''
        with self.lifecycle_lock:
            if self.thread_auto_bot is not None and self.thread_auto_bot.is_alive():
                raise RuntimeError("AutoBot is already running")
            self.is_terminated = False
            self.runtime = BotRuntime(
                self.cfg,
                self.args,
                self.is_disable_control,
                self.loop,
                KeyBoardController,
                GameWindowCapturor,
                StaticImageCapturor,
                HealthMonitor,
                Profiler,
            )
            self.runtime.prepare()
            self.kb = self.runtime.keyboard
            self.capture = self.runtime.capture
            self.health_monitor = self.runtime.health_monitor
            self.profiler = self.runtime.profiler

            # Reset all timers
            self.t_last_frame = time.time()
            self.t_watch_dog = time.time()
            self.t_last_teleport = time.time()
            self.t_last_attack = time.time()
            self.t_last_minimap_update = time.time()
            self.t_to_change_channel = time.time()

            # Start Auto Bot main thread
            self.thread_auto_bot = self.runtime.start_thread()
            self.is_first_frame = True

        logger.info("[MapleStoryAutoBot] Started")

    def pause(self):
        '''
        Terminate thread except main thread
        '''
        self.terminate_threads()

    def enable_viz(self):
        self.is_need_show_debug_window = True
        logger.debug("[enable_viz] is_show_debug_window = True")

    def disable_viz(self):
        self.is_need_show_debug_window = False
        logger.debug("[disable_viz] is_show_debug_window = False")

    def start_record(self):
        '''
        Start record
        '''
        # Prepare video writer if need to record
        if not self.is_show_debug_window:
            self.enable_viz()

        # Make sure video/ exist
        os.makedirs("video", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("video", f"{timestamp}.mp4")

        # Get video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
        self.video_writer = cv2.VideoWriter(path, fourcc, 10, (1296, 759))

        logger.info(f"[start_record] Record video to {path}")

    def stop_record(self):
        '''
        Stop Record
        '''
        self.video_writer = None
        logger.info("[stop_record] Stop recording")

    def get_player_location_by_nametag(self):
        '''
        Detects the player's location based on the nametag position in the game window.

        This function works by:
        - Extracting a vertical region of interest (ROI) where the nametag is expected.
        - Padding the ROI to avoid template matching edge issues.
        - Using template matching to locate the nametag, split into left and right halves
        to improve robustness against partial occlusion.
        - Selecting the best match (left or right) based on score and cache status.
        - Computing the player's center position by applying a fixed offset to the nametag.

        Returns:
            loc_player (tuple): The (x, y) coordinates of the player's estimated location.
        '''
        result = PlayerLocator(self.cfg).by_nametag(
            self.img_frame_gray,
            self.img_frame_debug,
            self.img_nametag,
            self.img_nametag_gray,
            self.loc_nametag,
            self.is_first_frame,
        )
        if result is None:
            return None
        self.loc_nametag = result.nametag
        return result.player

    def get_player_location_by_party_red_bar(self):
        '''
        get_player_location_by_party_red_bar
        '''
        return PlayerLocator(self.cfg).by_party_red_bar(
            self.img_frame,
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape,
        )

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map
        '''
        result = PlayerLocator(self.cfg).on_global_map(
            self.img_map,
            self.img_minimap,
            self.loc_player_minimap,
            self.img_route_debug,
        )
        self.loc_minimap_global = result.minimap_origin
        return result.player

    def get_nearest_color_code(self):
        '''
        Searches for the nearest color-coded action marker
        around the player on the route map.

        This function:
        - Scans each pixel in the search box to find nearest color code
        - Tracks the closest matching pixel using Manhattan distance (|dx| + |dy|).
        - Returns a dictionary containing the nearest matching
          pixel's position, color, action label, and distance.

        Returns:
            dict or None: Dictionary containing:
                - "pixel": (x, y) coordinate of the matched pixel
                - "color": matched RGB color tuple
                - "action": corresponding action string from config
                - "distance": Manhattan distance from player
            Returns None if no matching color is found within the region.
        '''
        nearest, nearest_up_down, bounds = find_nearest_actions(
            self.img_route,
            self.loc_player_global,
            self.cfg["route"]["search_range"],
            self.color_code,
            self.color_code_up_down,
        )
        x_min, y_min, _, _ = bounds

        # Debug
        draw_rectangle(
            self.img_route_debug,
            (x_min, y_min),
            (self.cfg["route"]["search_range"]*2,
             self.cfg["route"]["search_range"]*2),
            (0, 0, 255), "", text_height=0.4, thickness=1,
        )
        # Draw a straigt line from map_loc_player to color_code["pixel"]
        if nearest is not None:
            cv2.line(
                self.img_route_debug,
                self.loc_player_global, # start point
                nearest["pixel"],       # end point
                (0, 255, 0),            # green line
                1                       # thickness
            )
            # Print color code on debug image
            cv2.putText(
                self.img_frame_debug, f"Route Action: {nearest['command']}",
                (650, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )
            cv2.putText(
                self.img_frame_debug, f"Route Index: {self.idx_routes}",
                (650, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        if nearest_up_down is not None:
            cv2.putText(
                self.img_frame_debug, f"Route Action: {nearest_up_down['command']}",
                (650, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )
            cv2.line(
                self.img_route_debug,
                self.loc_player_global,  # start point
                nearest_up_down["pixel"],# end point
                (0, 0, 255),             # green line
                1                        # thickness
            )

        return nearest, nearest_up_down  # if not found return none

    def get_attack_range(self, is_left=True):
        '''
        get_attack_range
        '''
        analyzer = CombatAnalyzer(self.cfg, self.monsters_info)
        return analyzer.attack_range(
            self.loc_player,
            self.img_frame.shape,
            is_left,
        )

    def get_nearest_monster(self, is_left=True):
        '''
        Finds the nearest monster within the player's attack range.

        This function:
        - Defines an attack box relative to the player position,
            depending on the facing direction (`is_left`).
        - Iterates through all detected monsters and checks which ones overlap
          with the attack box.
        - Returns the closest valid monster that meets the overlap criteria.

        Args:
            is_left (bool): If True, assume the player is facing left;
                            adjusts attack box accordingly.
        Returns:
            dict or None: The nearest monster's info dict, or None if no valid match.
        '''

        analyzer = CombatAnalyzer(self.cfg, self.monsters_info)
        return analyzer.nearest_monster(
            self.monsters,
            self.loc_player,
            self.img_frame.shape,
            is_left,
        )

    def get_monsters_in_range(self, top_left, bottom_right):
        '''
        get_monsters_in_range
        '''
        detector = MonsterDetector(self.cfg, self.monsters_info)
        return detector.detect(
            self.img_frame,
            self.img_frame_debug,
            self.loc_player,
            top_left,
            bottom_right,
        )

    def get_img_frame(self):
        '''
        get_img_frame
        '''
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        result = normalize_game_frame(
            self.frame,
            self.cfg,
            skip_aspect_check=bool(self.args.test_image),
        )
        if result.error:
            if self.frame is None:
                logger.warning(result.error)
            else:
                logger.error(result.error)
            return None
        if result.was_resized:
            logger.info(
                "Resize game window frame from "
                f"{self.frame.shape[:2]} to canonical processing size "
                "(759, 1296)"
            )
        return result.image

    def is_player_stuck(self):
        """
        Checks whether the player is stuck (not moving)
        based on their global position on map.

        This function:
        - Compares the player's current position with their last known position
          tracked by the watchdog.
        - If the player has moved beyond a threshold (`watch_dog_range`),
          it resets the watchdog timer.
        - If the player hasn't moved and the elapsed time exceeds (`watch_dog_timeout`),
          it flags the player as stuck and resets the watchdog.

        Returns:
            bool: True if the player is stuck, False otherwise.
        """
        dx = abs(self.loc_player_global[0] - self.loc_watch_dog[0])
        dy = abs(self.loc_player_global[1] - self.loc_watch_dog[1])

        current_time = time.time()
        if dx + dy > self.cfg["watchdog"]["range"]:
            # Player moved, reset watchdog timer
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            return False

        dt = current_time - self.t_watch_dog
        if dt > self.cfg["watchdog"]["timeout"]:
            # watch dog idle for too long, player stuck
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            logger.warning(f"[is_player_stuck] Player stuck for {round(dt, 2)} seconds.")
            return True
        return False

    def screenshot_img_frame(self):
        '''
        Save self.img_frame
        '''
        if self.img_frame is None:
            logger.error("[screenshot_img_frame] Failed, game window is not available")
        else:
            screenshot(self.img_frame)

    def is_near_edge(self):
        '''
        Detects whether the player is near a teleport edge region

        This function:
        - Defines a rectangular search region around the player's current global location.
        - Scans for pixels matching a specific edge teleport color code within the region.
        - If matching pixels are found, it computes the average X position of those pixels.
        - Compares that average to the player's X position to determine whether the edge is on the left or right.

        Returns:
            str: One of:
                - "edge on left"
                - "edge on right"
                - "" (empty string if no edge is detected nearby)
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        h_trigger_box = self.cfg["edge_teleport"]["trigger_box_height"]
        w_trigger_box = self.cfg["edge_teleport"]["trigger_box_width"]
        x_min = max(0, x0 - w_trigger_box//2)
        x_max = min(w, x0 + w_trigger_box//2)
        y_min = max(0, y0 - h_trigger_box//2)
        y_max = min(h, y0 + h_trigger_box//2)

        # Debug: draw search box
        # draw_rectangle(
        #     self.img_route_debug,
        #     (x_min, y_min),
        #     (y_max - y_min, x_max - x_min),
        #     (0, 0, 255), "Edge Check", thickness=1, text_height=0.4
        # )

        # Find mask of matching pixels
        roi = self.img_route[y_min:y_max, x_min:x_max]
        mask = np.all(roi == self.cfg["edge_teleport"]["color_code"], axis=2)
        coords = np.column_stack(np.where(mask))

        # No edge pixel
        if coords.size == 0:
            return ""

        # Calculate mean position of matching pixels
        mean_x = np.mean(coords[:, 1])

        # Compare to roi center
        if mean_x < x0:
            return "edge on left"
        else:
            return "edge on right"

    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        self.fps = round(1.0 / (time.time() - self.t_last_frame))
        text_y_interval = 23
        text_y_start = 520
        dt_screenshot = time.time() - self.kb.t_last_screenshot
        h, w = self.frame.shape[:2]
        text_list = [
            f"FPS: {self.fps}",
            f"State: {self.fsm.state.name}",
            f"Resolution: {h}x{w}, Ratio: {round(w/h, 2)}",
            f"Press 'F1' to {'pause' if self.kb.is_enable else 'start'} Bot",
            f"Press 'F2' to save screenshot{' : Saved' if dt_screenshot < 0.7 else ''}",
             "Press 'F12' to quit"]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        # Draw attack box on debug window
        if self.cfg["bot"]["attack"] == "aoe_skill":
            x0, y0, x1, y1 = self.get_attack_range()
            draw_rectangle(
                self.img_frame_debug, (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range"
            )
        elif self.cfg["bot"]["attack"] == "directional":
            x0, y0, x1, y1 = self.get_attack_range(is_left=True)
            draw_rectangle(
                self.img_frame_debug, (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range(Left)"
            )
            x0, y0, x1, y1 = self.get_attack_range(is_left=False)
            draw_rectangle(
                self.img_frame_debug, (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range(Right)"
            )

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape[:2],
            (0, 0, 255), "minimap",thickness=2
        )

        # Don't draw minimap in patrol mode
        if self.cfg["bot"]["mode"] in ["patrol", "aux"]:
            return

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # Check if valid crop region
        if x1 <= x0 or y1 <= y0:
            return

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        mini_map_crop = cv2.resize(mini_map_crop,
                                (int(mini_map_crop.shape[1] * 3),
                                 int(mini_map_crop.shape[0] * 3)),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        h_frame, w_frame = self.img_frame_debug.shape[:2]
        x_paste = w_frame - w_crop - 10  # 10px margin from right
        y_paste = 70
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=2
        )

        # Draw HP/MP/EXP bar on debug window
        percent_bars = [self.health_monitor.hp_percent,
                      self.health_monitor.mp_percent,
                      self.health_monitor.exp_percent]
        for i, bar_name in enumerate(["HP", "MP", "EXP"]):
            x_s, y_s = (250, 90)
            # Print bar ratio on debug window
            cv2.putText(self.img_frame_debug,
                        f"{bar_name}: {percent_bars[i]:.1f}%",
                        (x_s, y_s + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # Draw bar on debug window
            x_s, y_s = (410, 73)
            x, y, w, h = self.health_monitor.loc_size_bars[i]
            self.img_frame_debug[y_s+30*i:y_s+h+30*i, x_s:x_s+w] = \
                self.img_frame[self.cfg["camera"]["y_end"]:, :][y:y+h, x:x+w]

        # Print command on screen
        cv2.putText(self.img_frame_debug, f"Cmd: {self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}",
                    (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        cv2.imshow("Game Window Debug",
            self.img_frame_debug[self.cfg["camera"]["y_start"]:
                                 self.cfg["camera"]["y_end"], :])
        # Update FPS timer
        self.t_last_frame = time.time()

    def ensure_is_in_party(self):
        '''
        ensure_is_in_party
        '''
        # open party window
        press_key(self.cfg["key"]["party"])

        try:
            # Wait party window to show up
            time.sleep(0.5)

            # Update image frame
            self.img_frame = self.get_img_frame()
            if self.img_frame is None:
                logger.error(
                    "[ensure_is_in_party] Cannot inspect party state because "
                    "the game frame is unavailable."
                )
                return False

            # Find the 'create party' button
            loc_enable, score_enable, _ = find_pattern_sqdiff(
                            self.img_frame, self.img_create_party_enable)

            lang = self.cfg["system"]["language"]
            thres = self.cfg['party_red_bar'][f'create_party_button_{lang}_thres']
            if score_enable < thres:
                logger.info(f"[ensure_is_in_party] Find party enable button({round(score_enable, 2)})")
                h, w = self.img_create_party_enable.shape[:2]
                click_in_game_window(self.cfg["game_window"]["title"],
                    (loc_enable[0] + w // 2, loc_enable[1] + h // 2)
                )
            else:
                logger.info("[ensure_is_in_party] Cannot find create party button."
                            "Maybe player already in party.")
            return True
        finally:
            # Always close the party window, including failed frame captures.
            press_key(self.cfg["key"]["party"])

    def channel_change(self):
        '''
        channel_change
        '''
        logger.info("[channel_change] Start")

        window_title = self.capture.window_title
        ui_coords = self.cfg["ui_coords"]
        click_in_game_window(window_title, ui_coords["menu"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel_confirm"])
        time.sleep(1)

        while self.get_login_button_location() is None:
            self.img_frame = self.get_img_frame()
            logger.info("Waiting for login button to show up...")
            time.sleep(3)
        time.sleep(3) # Wait screen to become brighter

        # Click login button
        click_in_game_window(window_title, self.get_login_button_location())
        time.sleep(2)

        # Click "Select Character"
        click_in_game_window(window_title, ui_coords["select_character"])
        time.sleep(5)

        self.kb.enable()
        self.kb.set_command("none none none")
        self.kb.release_all_key()

        self.ensure_is_in_party() # Make sure player is in party

        self.fsm.set_state(BOT_MODE_STATES[self.cfg["bot"]["mode"]])
        self.t_last_attack = time.time() # Update timer

    def terminate_threads(self):
        '''
        terminate all threads
        '''
        with self.lifecycle_lock:
            self.is_terminated = True
            if self.runtime is not None:
                self.runtime.stop()
        logger.info("[terminate_threads] Terminated all threads")

    def get_attack_direction(self, monster_left, monster_right):
        '''
        get_attack_direction
        '''
        # Compute distance for left
        distance_left = float('inf')
        if monster_left is not None:
            mx, my = monster_left["position"]
            mw, mh = monster_left["size"]
            center_left = (mx + mw // 2, my + mh // 2)
            distance_left = abs(center_left[0] - self.loc_player[0]) + \
                            abs(center_left[1] - self.loc_player[1])
        # Compute distance for right
        distance_right = float('inf')
        if monster_right is not None:
            mx, my = monster_right["position"]
            mw, mh = monster_right["size"]
            center_right = (mx + mw // 2, my + mh // 2)
            distance_right = abs(center_right[0] - self.loc_player[0]) + \
                            abs(center_right[1] - self.loc_player[1])
        # Choose attack direction and nearest monster
        attack_direction = None
        # nearest_monster = None

        # Additional validation: check if monster is actually on the correct side
        def is_monster_on_correct_side(monster, direction):
            if monster is None:
                return False
            mx, my = monster["position"]
            mw, mh = monster["size"]
            monster_center_x = mx + mw // 2
            player_x = self.loc_player[0]

            if direction == "left":
                return monster_center_x < player_x  # Monster should be left of player
            else:  # direction == "right"
                return monster_center_x > player_x  # Monster should be right of player

        # Only choose direction if there's a clear winner and monster is on correct side
        if monster_left is not None and monster_right is None and \
            is_monster_on_correct_side(monster_left, "left"):
            attack_direction = "left"
            # nearest_monster = monster_left
        elif monster_right is not None and monster_left is None and \
            is_monster_on_correct_side(monster_right, "right"):
            attack_direction = "right"
            # nearest_monster = monster_right
        elif monster_left is not None and monster_right is not None:
            # Both sides have monsters, check distance and side validation
            left_valid = is_monster_on_correct_side(monster_left, "left")
            right_valid = is_monster_on_correct_side(monster_right, "right")

            if left_valid and not right_valid:
                attack_direction = "left"
                # nearest_monster = monster_left
            elif right_valid and not left_valid:
                attack_direction = "right"
                # nearest_monster = monster_right
            elif left_valid and right_valid and distance_left < distance_right - 50:
                attack_direction = "left"
                # nearest_monster = monster_left
            elif left_valid and right_valid and distance_right < distance_left - 50:
                attack_direction = "right"
                # nearest_monster = monster_right
            # If both valid but distances too close, don't attack to avoid confusion

        # Debug attack direction selection
        if monster_left is not None or monster_right is not None:
            left_side_ok = is_monster_on_correct_side(monster_left, "left") if monster_left else False
            right_side_ok = is_monster_on_correct_side(monster_right, "right") if monster_right else False
            debug_text = f"L:{distance_left:.0f}({left_side_ok}) R:{distance_right:.0f}({right_side_ok}) Dir:{attack_direction}"
            cv2.putText(self.img_frame_debug, debug_text,
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return attack_direction

    def is_need_change_channel(self, loc_other_players):
        '''
        is_need_change_channel
        '''
        # Calculate center value
        xs = [x for (x, _) in loc_other_players]
        ys = [y for (_, y) in loc_other_players]
        if len(xs) == 0 or len(ys) == 0:
            return False
        center_x, center_y = (np.mean(xs), np.mean(ys))
        if np.isnan(center_x) or np.isnan(center_y):
            return False
        center = (int(np.mean(xs)), int(np.mean(ys)))
        #logger.info(f"[is_need_change_channel] Center of mass = {center}")

        # Change channel
        mode = self.cfg["channel_change"]["mode"]
        if mode == "true":
            logger.warning("[is_need_change_channel] Player detected, immediately change channel.")
            return True
        elif mode == "pixel":
            if self.red_dot_center_prev is None:
                self.red_dot_center_prev = center
            else:
                dx = abs(center[0] - self.red_dot_center_prev[0])
                dy = abs(center[1] - self.red_dot_center_prev[1])
                total = dx + dy
                logger.debug(f"[is_need_change_channel] Movement dx={dx}, dy={dy}, total={total}")
                thres = self.cfg["channel_change"]["other_player_move_thres"]
                if total > thres:
                    logger.warning(f"Other player movement > {thres} pixel detected. "
                                "Trigger channel change.")
                    return True
        else:
            logger.error(f"[is_need_change_channel] Unsupported mode: {mode}")

        return False

    def is_time_to_change_channel(self):
        '''
        is_time_to_change_channel
        '''
        if not self.cfg["scheduled_channel_switching"]["enable"]:
            return False
        dt = time.time() - self.t_to_change_channel
        if dt > self.cfg["scheduled_channel_switching"]["interval_seconds"]:
            self.t_to_change_channel = time.time()
            return True
        return False

    def get_login_button_location(self):
        '''
        get_login_button_location
        '''
        # Extract the region where the login button should appear
        x0, y0 = self.cfg["ui_coords"]["login_button_top_left"]
        x1, y1 = self.cfg["ui_coords"]["login_button_bottom_right"]
        img_roi = self.img_frame[y0:y1, x0:x1]

        # Find the 'login' button
        loc, score, _ = find_pattern_sqdiff(
                        img_roi, self.img_login_button)
        if score < self.cfg["ui_coords"]["login_button_thres"]:
            h, w = self.img_login_button.shape[:2]
            logger.info(f"[get_login_button_location] Found login button with score({score})")
            return (x0 + loc[0] + w // 2, y0 + loc[1] + h // 2)
        else:
            return None

    def update_cmd_by_route(self):
        # get color code from img_route
        color_code, color_code_up_down = self.get_nearest_color_code()
        # Use color_code and color_code_up_down to complement each other
        # To prevent character stuck at the end of ladder, we use two color color pixels
        # and let them complement with each other, to ensure smoothy ladder climbing
        if color_code and color_code_up_down:
            if color_code["distance"] < color_code_up_down["distance"]:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
                _, cmd, _ = color_code_up_down["command"].split()
                if self.cmd_move_y == "none" and self.is_on_ladder:
                    self.cmd_move_y = cmd # only complement cmd_move_y when player is on ladder
            else:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()
                cmd, _, _ = color_code["command"].split()
                if self.cmd_move_x == "none" and self.is_on_ladder:
                    self.cmd_move_x = cmd # only complement cmd_move_x when player is on ladder
        elif color_code:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
        elif color_code_up_down:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()

        # teleport away from edge to avoid falling off cliff
        if self.is_near_edge() and \
            time.time() - self.t_last_teleport > self.cfg["teleport"]["cooldown"]:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # Use teleport while walking
        if self.cfg['teleport']['is_use_teleport_to_walk'] and \
            time.time() - self.t_last_teleport > self.cfg['teleport']['cooldown']:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # replace teleport to jump if user doesn't set teleport key
        if self.cfg["key"]["teleport"] == "" and self.cmd_action == "teleport":
            self.cmd_action = "jump"

    def update_cmd_by_mob_detection(self):
        # Get monster search box
        margin = self.cfg["monster_detect"]["search_box_margin"]
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2 + margin
            dy = self.cfg["aoe_skill"]["range_y"] // 2 + margin
            cooldown = self.cfg["aoe_skill"]["cooldown"]
        elif self.cfg["bot"]["attack"] == "directional":
            dx = self.cfg["directional_attack"]["range_x"] + margin
            dy = self.cfg["directional_attack"]["range_y"] + margin
            cooldown = self.cfg["directional_attack"]["cooldown"]
        else:
            raise RuntimeError(f"Unsupported attack mode: {self.cfg['bot']['attack']}")
        x0 = max(0                      , self.loc_player[0] - dx)
        x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
        y0 = max(0                      , self.loc_player[1] - dy)
        y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        # Get monsters in the search box
        self.monsters = self.get_monsters_in_range((x0, y0), (x1, y1))

        # Check if no mob to attack
        if len(self.monsters) == 0:
            return

        # Update attack command
        if self.cfg["bot"]["attack"] == "aoe_skill":
            if time.time() - self.t_last_attack > cooldown:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()

        elif self.cfg["bot"]["attack"] == "directional":
            # Get nearest monster to player
            monster_left  = self.get_nearest_monster(is_left = True)
            monster_right = self.get_nearest_monster(is_left = False)
            # Determine attack direction
            attack_direction = self.get_attack_direction(monster_left, monster_right)
            # Attack Command
            if time.time() - self.t_last_attack > cooldown and attack_direction is not None:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()
                # Set up attack direction
                self.cmd_move_x = attack_direction

    def update_cmd_by_random(self):
        '''
        update_cmd_by_random - pick a random action except 'up' and teleport command
        '''
        self.cmd_move_x = random.choice(["left", "right", "none"])
        self.cmd_move_y = random.choice(["down", "none"])
        self.cmd_action = random.choice(["jump", "none"])
        logger.warning("[update_cmd_by_random]"\
                    f"{self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}")

    def check_reach_goal(self):
        if self.cmd_action == "goal":
            # Switch to next route map
            self.idx_routes = (self.idx_routes+1)%len(self.img_routes)
            logger.debug(f"Change to new route:{self.idx_routes}")

    def run_once(self):
        '''
        Process one game window frame
        '''
        # Start profiler for performance debugging
        self.profiler.start()

        # Check if need viz window
        self.is_show_debug_window = self.is_need_show_debug_window
        if not self.is_show_debug_window:
            self.img_frame_debug = None
            self.img_route_debug = None

        ###########################
        ### Image Preprocessing ###
        ###########################
        # Get game window frame
        img_frame = self.get_img_frame()
        if img_frame is None:
            if not is_mac():
                activate_game_window(self.capture.window_title)
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Image for debug viz
        if self.is_show_debug_window:
            self.img_frame_debug = self.img_frame.copy()

        # Get current route image
        if self.cfg["bot"]["mode"] == "normal":
            self.img_route = self.img_routes[self.idx_routes]
            if self.is_show_debug_window:
                self.img_route_debug = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)

        self.profiler.mark("Image Preprocessing")

        ###################
        ### Get Minimap ###
        ###################
        # Get minimap coordinate and size on game window
        minimap_result = get_minimap_loc_size(self.img_frame)
        if minimap_result is None:
            if time.time() - self.t_last_minimap_update > 30:
                # Unable to get minimap for 30 seconds -> assume it's login screen
                loc_login_button = self.get_login_button_location()
                if loc_login_button:
                    logger.info("Found login button on screen. Proceed to login.")
                    click_in_game_window(self.cfg["game_window"]["title"],
                                         loc_login_button)
                    time.sleep(3)
                    click_in_game_window(self.cfg["game_window"]["title"],
                                         self.cfg["ui_coords"]["select_character"])
                    time.sleep(2)
        else:
            x, y, w, h = minimap_result
            # Shrink minimap boardary by one pixel to avoid pixel leaking to minimap
            x += 1
            y += 1
            w -= 2
            h -= 2
            # update minimap image
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y+h, x:x+w]
            self.t_last_minimap_update = time.time()

        self.profiler.mark("Get Minimap Location and Size")

        # Update health monitor with current frame
        self.health_monitor.update_frame(self.img_frame[self.cfg["camera"]["y_end"]:, :])

        #################################
        ### Player Location Detection ###
        #################################
        # Get player location in game window
        if self.cfg["nametag"]["enable"]:
            loc_player = self.get_player_location_by_nametag()
        else:
            loc_player, loc_party_red_bar = self.get_player_location_by_party_red_bar()
            if loc_party_red_bar is not None:
                self.loc_party_red_bar = loc_party_red_bar

        # Update player location
        if loc_player is not None:
            # Check if character is on ladder
            dx = abs(loc_player[0] - self.loc_player[0])
            dy = abs(loc_player[1] - self.loc_player[1])
            if self.is_on_ladder:
                if dx > 3: # Leave ladder if there is horizontal move
                    self.is_on_ladder = False
            else:
                if dx < 3 and dy != 0:
                    self.is_on_ladder = True
            # logger.info((self.is_on_ladder, dx, dy))
            # Update player location
            self.loc_player = loc_player

        # Draw player center for debugging
        cv2.circle(self.img_frame_debug,
                self.loc_player, radius=3,
                color=(0, 0, 255), thickness=-1)

        # Get player location on minimap
        loc_player_minimap = get_player_location_on_minimap(
                                self.img_minimap,
                                minimap_player_color=self.cfg["minimap"]["player_color"])
        if loc_player_minimap:
            self.loc_player_minimap = loc_player_minimap

        # Get other player location on minimap
        loc_other_players = get_all_other_player_locations_on_minimap(
                                self.img_minimap,
                                self.cfg['minimap']['other_player_color'])
        # Debug
        # if self.is_first_frame:
        #     logger.info("Running minimap color analysis...")
        #     debug_minimap_colors(self.img_minimap, other_player_color)

        # Get player location on global map
        if self.cfg["bot"]["mode"] in ["patrol", "aux"]:
            self.loc_player_global = self.loc_player_minimap
        else:
            self.loc_player_global = self.get_player_location_on_global_map()

        self.profiler.mark("Player Location Detection")

        ######################
        ### Change Channel ###
        ######################
        if self.cfg['channel_change']['enable'] and \
            self.is_need_change_channel(loc_other_players):
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            self.kb.disable()
            time.sleep(1)
            self.channel_change()
            self.red_dot_center_prev = None
            return 0

        if self.is_time_to_change_channel():
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            self.kb.disable()
            time.sleep(1)
            self.channel_change()
            return 0

        self.profiler.mark("Change Channel")

        #######################
        ### Attack WatchDog ###
        ####################### Check if last attack is timeout
        dt = time.time() - self.t_last_attack
        if self.cfg['bot']['mode'] == 'normal' and \
            dt > self.cfg["watchdog"]["last_attack_timeout"]:
            logger.info(f"[Attack Timeout] Last attack timeout for {round(dt, 2)} seconds")
            cfg_action = self.cfg["watchdog"]["last_attack_timeout_action"]
            if cfg_action == "change_channel":
                logger.info("[Attack Timeout] Change channel!")
                self.channel_change()
            elif cfg_action == "go_home":
                logger.info("[Attack Timeout] Return home!")
                press_key(self.cfg["key"]["return_home"])
                # Terminate Autobot
                self.is_terminated = True
                self.kb.is_terminated = True
            else:
                logger.info(f"Unsupported timeout mode: {cfg_action}")

        self.profiler.mark("Attack WatchDog")

        ######################
        ### State Behavior ###
        ######################
        self.fsm.run_frame()

        self.is_first_frame = False

        self.profiler.mark("State per-frame behavior")

        #####################
        ### Debug Windows ###
        #####################
        # Don't show debug window to save system resource
        if not self.is_show_debug_window:
            return 0 # frame done

        # Print text on debug image
        self.update_info_on_img_frame_debug()

        # Save debug window to video
        if self.video_writer:
            self.video_writer.write(self.img_frame_debug)

        # Resize img_route_debug for better visualization
        if self.cfg["bot"]["mode"] == "normal":
            self.img_route_debug = cv2.resize(
                        self.img_route_debug, (0, 0),
                        fx=self.cfg["minimap"]["debug_window_upscale"],
                        fy=self.cfg["minimap"]["debug_window_upscale"],
                        interpolation=cv2.INTER_NEAREST)

        self.profiler.mark("Debug Window Show")

        # Update FPS timer
        self.t_last_frame = time.time()

        # Check FPS, TODO: too verbose, only print if many frames has high latency
        # if self.fps < 5:
        #     logger.warning(f"FPS({self.fps}) is too low, AutoBot cannot run properly!")

        # Print profiler result
        if self.cfg["profiler"]["enable"] and \
            self.profiler.total_frames % self.cfg["profiler"]["print_frequency"] == 0:
            logger.info('\n' + self.profiler.report())

        return 0 # frame done

    def loop(self):
        '''
        Auto Bot main loop
        Only run when call autobot from UI framework and AutoBotController
        '''
        # Make sure player is in party
        if not is_mac() and self.args.test_image == '':
            activate_game_window(self.capture.window_title)
            time.sleep(0.3)
            self.ensure_is_in_party()

        while not self.kb.is_terminated:

            t_start = time.time()

            # Process one game window frame
            self.is_frame_done = False
            ret = self.run_once()

            # Only proceed if the frame is valid
            if ret == 0:
                # Draw image on debug window
                if self.is_show_debug_window and self.is_ui and \
                        self.img_frame_debug is not None:
                    img_frame_debug_emit = self.img_frame_debug[
                        self.cfg["camera"]["y_start"]:
                        self.cfg["camera"]["y_end"], :].copy()
                    self.image_debug_signal.emit(img_frame_debug_emit)
                    img_route_debug_emit = None
                    if self.img_route_debug is not None:
                        img_route_debug_emit = self.img_route_debug.copy()
                    self.route_map_viz_signal.emit(img_route_debug_emit)
            self.is_frame_done = True

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / self.cfg["system"]["fps_limit_main"]
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

def main(args):
    '''
    This main function works as a fake autoBotController
    This function will only be called when the using terminal to
    run this script
    '''
    #####################
    ### Init Auto Bot ###
    #####################
    try:
        mapleStoryAutoBot = MapleStoryAutoBot(args)
    except Exception as e:
        logger.error(f"MapleStoryAutoBot Init failed: {e}")
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Init Successfully")

    ####################
    ### Apply Config ###
    ####################
    # Load defautl yaml config
    cfg = load_yaml("config/config_default.yaml")
    # Override with platform config
    if is_mac():
        cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
    # Override with user customized config
    cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))
    # Dump config to log for debugging
    logger.debug(yaml.dump(cfg, sort_keys=False,
                 indent=2, default_flow_style=False))
    # autoBot load config
    mapleStoryAutoBot.load_config(cfg)

    #####################
    ### Start AutoBot ###
    #####################
    try:
        mapleStoryAutoBot.start() # Start all threads in autoBot
    except Exception as e:
        logger.error(f"MapleStoryAutoBot start failed: {e}")
        mapleStoryAutoBot.terminate_threads() # Terminate all threads
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Start Successfully")

    # Start record game window for debugging
    if args.record:
        mapleStoryAutoBot.start_record()

    kb_listener = KeyBoardListener(is_autobot=True)
    kb_listener.register_func_key_handler('f1', mapleStoryAutoBot.kb.toggle_enable)
    kb_listener.register_func_key_handler('f2', mapleStoryAutoBot.screenshot_img_frame)
    kb_listener.register_func_key_handler('f12', mapleStoryAutoBot.terminate_threads)

    # While loop
    while not mapleStoryAutoBot.is_terminated:
        # Show debug image on window
        if mapleStoryAutoBot.is_frame_done:
            if mapleStoryAutoBot.img_frame_debug is not None:
                cv2.imshow("Game Window Debug",
                    mapleStoryAutoBot.img_frame_debug[
                        mapleStoryAutoBot.cfg["camera"]["y_start"]:
                        mapleStoryAutoBot.cfg["camera"]["y_end"], :])

            if mapleStoryAutoBot.img_route_debug is not None:
                cv2.imshow("Route Map Debug", mapleStoryAutoBot.img_route_debug)

        cv2.waitKey(1)

        time.sleep(0.01)

    #########################
    ### Terminate AutoBot ###
    #########################
    mapleStoryAutoBot.terminate_threads() # Terminate all threads

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_control',
        action='store_true',
        help='Disable simulated keyboard input'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    parser.add_argument(
        '--debug',
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        '--record',
        action="store_true",
        help="Record debug window"
    )

    parser.add_argument(
        '--disable_viz',
        action="store_true",
        help="Disable viz debug window"
    )

    parser.add_argument(
        '--test_image',
        default="",
        help="Pass in image in test/XXX.png"
    )

    args = parser.parse_args()
    args.is_ui = False # Always set False for command line

    # Set logger level
    if args.debug:
        logger.set_level(logging.DEBUG)

    main(args)
