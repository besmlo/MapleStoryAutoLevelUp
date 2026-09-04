'''
Auto generate route map
'''
# Standard import
import os
import re
import shutil
import time

import cv2

# CV import
import numpy as np

from src.engine.MapScanner import MapScanner
from src.input.GameWindowCapturor import GameWindowCapturor
from src.input.KeyBoardListener import KeyBoardListener
from src.utils.common import (
    draw_rectangle,
    find_pattern_sqdiff,
    get_minimap_loc_size,
    get_player_location_on_minimap,
    is_mac,
    load_image,
    load_yaml,
    override_cfg,
    screenshot,
)

# local import
from src.utils.logger import logger


class RouteRecorderEngine:
    '''
    Route recorder
    '''
    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        self.fps = round(1.0 / (time.time() - self.t_last_frame))
        text_y_interval = 23
        text_y_start = 550
        dt_screenshot = time.time() - self.kb.t_func_key[1]
        dt_save_route = time.time() - self.kb.t_func_key[2]
        dt_save_map = time.time() - self.kb.t_func_key[3]
        text_list = [
            f"FPS: {self.fps}",
            f"Press 'F1' to {'pause' if self.is_enable else 'start'} route record",
            f"Press 'F2' to save screenshot{' : Saved' if dt_screenshot < 0.7 else ''}",
            f"Press 'F3' to save route{' : Saved' if dt_save_route < 0.7 else ''}",
            f"Press 'F4' to save map{' : Saved' if dt_save_map < 0.7 else ''}",
        ]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape[:2],
            (0, 0, 255), "minimap",thickness=1
        )

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        mini_map_crop = cv2.resize(mini_map_crop,
                                (int(mini_map_crop.shape[1] * 3),
                                 int(mini_map_crop.shape[0] * 3)),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        _, w_frame = self.img_frame_debug.shape[:2]
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

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        if self.show_debug_windows:
            cv2.imshow("Game Window Debug",
                       self.img_frame_debug[self.cfg["camera"]["y_start"]:
                                            self.cfg["camera"]["y_end"], :])
        # Update FPS timer
        self.t_last_frame = time.time()

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map
        '''
        self.loc_minimap_global, score, _ = find_pattern_sqdiff(
                                        self.img_map,
                                        self.img_minimap)
        loc_player_global = (
            self.loc_minimap_global[0] + self.loc_player_minimap[0],
            self.loc_minimap_global[1] + self.loc_player_minimap[1]
        )

        # Draw local minimap rectangle
        camera_bottom_right = (
            self.loc_minimap_global[0] + self.img_minimap.shape[1],
            self.loc_minimap_global[1] + self.img_minimap.shape[0]
        )
        cv2.rectangle(self.img_route_debug, self.loc_minimap_global,
                      camera_bottom_right, (0, 255, 255), 1)
        cv2.putText(
            self.img_route_debug,
            f"Minimap,score({round(score, 2)})",
            (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 255), 1
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=2,
                   color=(0, 255, 255), thickness=-1)

        return loc_player_global

    def replace_color_on_map(self, lower_hsv, upper_hsv, replace_color=(0, 0, 0)):
        '''
        Replace pixels in self.img_map that fall within the given HSV range
        and are part of a connected component with area > 15.
        '''
        self.img_map = self.map_scanner.replace_hsv_components(
            self.img_map,
            lower_hsv,
            upper_hsv,
            replace_color,
        )

    def get_img_frame(self):
        '''
        get_img_frame
        '''
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        # Make sure the window ratio is as expected
        if self.cfg["game_window"]["size"] != self.frame.shape[:2]:
            text = f"Unexpeted window size: {self.frame.shape[:2]} "\
                    f"(expect {self.cfg['game_window']['size']})\n"
            text += "Please use windowed mode & smallest resolution."
            logger.error(text)
            return

        # Resize raw frame to (1296, 759)
        return cv2.resize(self.frame, (1296, 759),
                   interpolation=cv2.INTER_NEAREST)

    def __init__(self, args, cfg=None, capture=None):
        '''
        Init MapleStoryBot
        '''
        self.args = args # User arguments
        self.show_debug_windows = getattr(args, "show_debug_windows", True)
        self.idx_routes = 0 # Index of route map
        self.fps = 0 # Frame per second
        self.is_first_frame = True # first frame flag
        self.is_enable = getattr(args, "start_recording", True)
        # Coordinate (top-left coordinate)
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_player_global_last = None # playeer location on global map last frame
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = None # minimap on game screen
        self.img_map = None # map
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_last_draw_blob = time.time() # Last draw blob timer

        if cfg is None:
            # Load default yaml config
            cfg = load_yaml("config/config_default.yaml")
            # Override with platform config
            if is_mac():
                cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
            # Override with user customized config
            cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))
        self.cfg = cfg

        # Parse color_code
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }
        color_code_up_down = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code_up_down"].items()
        }
        self.color_code.update(color_code_up_down) # Combine both dictionaries
        self.map_scanner = MapScanner(
            cfg["route_recoder"]["map_padding"],
            self.color_code,
        )

        self.fps_limit = self.cfg["system"]["fps_limit_route_recorder"]

        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.new_map):
            raise ValueError(
                "Map ID may contain only letters, numbers, underscores, and hyphens"
            )

        # Check/create the new map directory.
        map_dir = os.path.join("minimaps", args.new_map)
        if os.path.exists(map_dir):
            replace_existing = getattr(args, "replace_existing", None)
            if replace_existing is None:
                user_input = input(
                    f"[Warning] Directory '{map_dir}' already exists. "
                    "Replace it? (y/n): "
                ).strip().lower()
                replace_existing = user_input == 'y'
            if replace_existing:
                shutil.rmtree(map_dir)  # Delete existing directory
                logger.info(f"Removed existing directory: {map_dir}")
            else:
                raise FileExistsError(f"Map directory already exists: {map_dir}")
        os.makedirs(map_dir) # Create new map directory
        self.map_dir = map_dir
        logger.info(f"Created new directory: {map_dir}")

        # Load exist map
        if self.args.map != '':
            self.img_map = load_image(f"{self.args.map}")

        self.kb = None
        self.capture = None
        try:
            # Start keyboard listener thread
            self.kb = KeyBoardListener(
                self.cfg,
                is_autobot=False,
                capture_function_keys=self.show_debug_windows,
            )

            # Start game window capturing thread
            logger.info(
                "Waiting for game window to activate, please click on game window"
            )
            self.capture = capture or GameWindowCapturor(self.cfg)
        except Exception:
            self.stop()
            raise

    def set_recording(self, enabled):
        self.is_enable = enabled
        logger.info(f"[RouteRecorder] Recording: {enabled}")

    def finish_route(self):
        if self.img_route is None:
            raise RuntimeError("Scan the minimap before saving a route")
        goal_rgb = next(
            color
            for color, metadata in self.color_code.items()
            if metadata["command"].split()[-1] == "goal"
        )
        cv2.circle(
            self.img_route,
            self.loc_player_global,
            radius=2,
            color=goal_rgb[::-1],
            thickness=-1,
        )
        out_path = os.path.join(self.map_dir, f"route{self.idx_routes+1}.png")
        if not cv2.imwrite(out_path, self.img_route):
            raise OSError(f"Unable to save route image: {out_path}")
        self.idx_routes += 1
        self.img_route = self.img_map.copy()
        self.loc_player_global_last = None
        logger.info(f"Save route image to {out_path}")
        return out_path

    def save_map(self):
        if self.img_map is None:
            raise RuntimeError("Scan the minimap before saving the map")
        out_path = os.path.join(self.map_dir, "map.png")
        if not cv2.imwrite(out_path, self.img_map):
            raise OSError(f"Unable to save map image: {out_path}")
        logger.info(f"Save map image to {out_path}")
        return out_path

    def stop(self):
        if self.kb is not None:
            self.kb.stop()
            self.kb = None
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        if self.show_debug_windows:
            cv2.destroyAllWindows()
        logger.info("[RouteRecorder] Terminated")

    def ensure_img_map_capacity(self, x, y, h, w):
        '''
        Ensure that self.img_map is large enough to contain the region defined by (x, y, h, w).
        Always add at least "map_padding" when expanding in any direction.
        '''
        expanded = self.map_scanner.ensure_capacity(
            self.img_map,
            self.img_route,
            (x, y),
            (h, w),
        )
        self.img_map = expanded.image
        self.img_route = expanded.route
        self.loc_minimap_global = expanded.origin

    def remove_color_code_pixels(self, img):
        """
        Set all pixels in self.img_map to black if they match any color in color_code (assumed RGB).
        """
        return self.map_scanner.remove_route_colors(img)

    def update_minimap(self):
        '''
        update_minimap
        '''

    def run_once(self):
        '''
        Process with one game window frame
        '''
        # Get lastest game screen frame buffer
        img_frame = self.get_img_frame()
        if img_frame is None:
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Get minimap from game window
        if self.is_first_frame:
            x, y, w, h = get_minimap_loc_size(self.img_frame)
            # Discard 1 pixel boundary of the minimap
            x += 1
            y += 1
            w -= 2
            h -= 2
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y+h, x:x+w]
        else:
            x, y = self.loc_minimap
            h, w = self.img_minimap.shape[:2]
            self.img_minimap = self.img_frame[y:y+h, x:x+w]

        # Replace black pixels (0, 0, 0) with (1, 1, 1)
        black_mask = np.all(self.img_minimap == [0, 0, 0], axis=-1)
        self.img_minimap[black_mask] = [1, 1, 1]

        # Get player location on minimap
        loc_player_minimap = get_player_location_on_minimap(self.img_minimap)
        if loc_player_minimap:
            self.loc_player_minimap = loc_player_minimap

        # Update map
        if self.is_first_frame:
            self.img_map, self.img_route = self.map_scanner.initialize(
                self.img_minimap,
                self.img_map,
            )
            self.img_route_debug = self.img_route.copy()

        else:
            self.loc_minimap_global, _ = self.map_scanner.locate(
                self.img_map,
                self.img_minimap,
            )
            x, y = self.loc_minimap_global
            h, w = self.img_minimap.shape[:2]
            # Ensure img_map is big enough to fit the newly explored region
            self.ensure_img_map_capacity(x, y, h, w)
            x, y = self.loc_minimap_global

            # Don't copy pixel near player
            player_yellow_dot_radius = 5
            px, py = self.loc_player_minimap
            h, w = self.img_minimap.shape[:2]
            x_min = max(0, px - player_yellow_dot_radius)
            x_max = min(w, px + player_yellow_dot_radius)
            y_min = max(0, py - player_yellow_dot_radius)
            y_max = min(h, py + player_yellow_dot_radius)
            # Apply the black color mask to mask player yellow dot
            self.img_minimap[y_min:y_max, x_min:x_max] = (0, 0, 0)

            # Update map
            if self.args.map == '':
                self.img_map = self.map_scanner.merge_unseen(
                    self.img_map,
                    self.img_minimap,
                    (x, y),
                )

            # Replace other player "red" dot to black on map
            self.replace_color_on_map((0, 78, 78),
                                      (5, 100, 100))

        if self.show_debug_windows:
            cv2.imshow("Map", self.img_map)
        self.img_route_debug = self.img_route.copy()

        # Get player location on global map
        self.loc_player_global = self.get_player_location_on_global_map()

        # Determine which color code to use based on user input
        action = ""
        is_draw_blob = False
        key_press = self.kb.key_pressing
        if "space" in key_press:
            if "left" in key_press:
                action = "left none jump"
            elif "right" in key_press:
                action = "right none jump"
            elif "down" in key_press:
                action = "none down jump"
            else:
                action = "none none jump"
            is_draw_blob = True
        elif "e" in key_press: # Teleport skill
            if "left" in key_press:
                action = "left none teleport"
            elif "right" in key_press:
                action = "right none teleport"
            elif "down" in key_press:
                action = "none down teleport"
            elif "up" in key_press:
                action = "none up teleport"
            else:
                action = ""
            is_draw_blob = True
        elif "up" in key_press:
            action = "none up none"
        elif "down" in key_press:
            action = "none down none"
        elif "left" in key_press:
            action = "left none none"
        elif "right" in key_press:
            action = "right none none"
        else:
            action = ""

        # Check if need to change route
        if self.kb.is_pressed_func_key[2]: # 'F3' is pressed
            action = "none none goal"
            is_draw_blob = True
            self.kb.is_pressed_func_key[2] = False
        elif self.kb.is_pressed_func_key[0]: # 'F1' is pressed
            self.is_enable = not self.is_enable
            logger.info(f"User press F1, is_enable = {self.is_enable}")
            self.kb.is_pressed_func_key[0] = False

        # Update route image
        if self.is_enable and action != "":
            # Get color from action
            dict_action_to_color = {v: k for k, v in self.color_code.items()}
            color_rgb = dict_action_to_color.get(action, None)
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Draw a line from the last position to the current one (if available)
            px, py = self.loc_player_global
            if is_draw_blob:
                dt = time.time() - self.t_last_draw_blob
                if dt > self.cfg["route_recoder"]["blob_cooldown"]:
                    # Draw a small filled circle at current position
                    cv2.circle(self.img_route,
                            (px, py),
                            radius=2,
                            color=color_bgr,
                            thickness=-1)  # filled circle
                    self.t_last_draw_blob = time.time()
                    self.loc_player_global_last = None
            else:
                if self.loc_player_global_last is None:
                    px_last, py_last = self.loc_player_global
                else:
                    px_last, py_last = self.loc_player_global_last
                cv2.line(self.img_route,
                        (px_last, py_last),
                        (px     , py),
                        color=color_bgr,
                        thickness=1)
                self.loc_player_global_last = self.loc_player_global

        # Save route image if goal is drawn
        if action == "none none goal":
            self.finish_route()

        # Save img_map to map.png
        if self.kb.is_pressed_func_key[3]: # 'F4' is pressed
            self.save_map()
            self.kb.is_pressed_func_key[3] = False

        #####################
        ### Debug Windows ###
        #####################
        # Print text on debug image
        self.update_info_on_img_frame_debug()

        # Show debug image on window
        self.update_img_frame_debug()

        # Check if need to save screenshot
        if self.kb.is_pressed_func_key[1]: # 'F2' is pressed
            screenshot(self.img_frame)
            self.kb.is_pressed_func_key[1] = False

        # Resize img_route_debug for better visualization
        self.img_route_debug = cv2.resize(
                    self.img_route_debug, (0, 0),
                    fx=self.cfg["minimap"]["debug_window_upscale"],
                    fy=self.cfg["minimap"]["debug_window_upscale"],
                    interpolation=cv2.INTER_NEAREST)
        if self.show_debug_windows:
            cv2.imshow("Route Map Debug", self.img_route_debug)

        # Enable cached location since second frame
        self.is_first_frame = False
