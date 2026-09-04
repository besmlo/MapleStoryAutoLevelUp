"""Monster detection isolated from bot navigation and keyboard control."""

import cv2
import numpy as np

from src.utils.common import draw_rectangle, nms
from src.utils.logger import logger


class MonsterDetector:
    def __init__(self, cfg, monster_templates):
        self.cfg = cfg
        self.monster_templates = monster_templates

    def detect(
        self,
        frame,
        debug_frame,
        player_location,
        top_left,
        bottom_right,
    ):
        x0, y0 = top_left
        x1, y1 = bottom_right
        image_roi = frame[y0:y1, x0:x1]
        character_bounds = self._character_bounds(
            image_roi,
            player_location,
            top_left,
        )

        mode = self.cfg["monster_detect"]["mode"]
        if self.cfg["bot"]["mode"] == "patrol":
            monsters = []
        elif mode == "template_free":
            has_templates = any(self.monster_templates.values())
            monsters = (
                self._detect_template_free(
                    image_roi,
                    debug_frame,
                    character_bounds,
                    top_left,
                )
                if has_templates
                else []
            )
        elif mode in {"contour_only", "grayscale", "color"}:
            monsters = self._detect_from_templates(
                image_roi,
                character_bounds,
                top_left,
                mode,
            )
            if monsters is None:
                return []
        else:
            logger.error(f"Unexpected monster detection mode: {mode}")
            return []

        monsters = nms(monsters, iou_threshold=0.4)
        if self.cfg["monster_detect"]["with_enemy_hp_bar"]:
            monsters.extend(self._detect_health_bars(image_roi, top_left))
        self._draw_debug(debug_frame, monsters, top_left, bottom_right)
        return monsters

    def _character_bounds(self, image_roi, player_location, top_left):
        player_x = player_location[0] - top_left[0]
        player_y = player_location[1] - top_left[1]
        half_width = self.cfg["character"]["width"] // 2
        half_height = self.cfg["character"]["height"] // 2
        return (
            max(0, player_x - half_width),
            max(0, player_y - half_height),
            min(image_roi.shape[1], player_x + half_width),
            min(image_roi.shape[0], player_y + half_height),
        )

    def _detect_template_free(
        self,
        image_roi,
        debug_frame,
        character_bounds,
        top_left,
    ):
        char_x_min, char_y_min, char_x_max, char_y_max = character_bounds
        black_mask = (
            np.all(image_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
        )
        black_mask[char_y_min:char_y_max, char_x_min:char_x_max] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

        draw_rectangle(
            debug_frame,
            (char_x_min + top_left[0], char_y_min + top_left[1]),
            (
                self.cfg["character"]["height"],
                self.cfg["character"]["width"],
            ),
            (255, 0, 0),
            "Character Box",
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            closed_mask,
            connectivity=8,
        )
        monsters = []
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            if area > 1000:
                monsters.append(
                    {
                        "name": "",
                        "position": (top_left[0] + x, top_left[1] + y),
                        "size": (height, width),
                        "score": 1.0,
                    }
                )
        return monsters

    def _detect_from_templates(
        self,
        image_roi,
        character_bounds,
        top_left,
        mode,
    ):
        monsters = []
        for monster_name, templates in self.monster_templates.items():
            for image, mask in templates:
                matches = self._match_template(
                    image_roi,
                    image,
                    mask,
                    character_bounds,
                    mode,
                )
                if matches is None:
                    return None
                height, width = image.shape[:2]
                for point, score in matches:
                    monsters.append(
                        {
                            "name": monster_name,
                            "position": (
                                point[0] + top_left[0],
                                point[1] + top_left[1],
                            ),
                            "size": (height, width),
                            "score": score,
                        }
                    )
        return monsters

    def _match_template(
        self,
        image_roi,
        template,
        template_mask,
        character_bounds,
        mode,
    ):
        if mode == "contour_only":
            template_input = (
                np.all(template == [0, 0, 0], axis=2).astype(np.uint8) * 255
            )
            roi_input = (
                np.all(image_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
            )
            x_min, y_min, x_max, y_max = character_bounds
            roi_input[y_min:y_max, x_min:x_max] = 0
            blur = self.cfg["monster_detect"]["contour_blur"]
            template_input = cv2.GaussianBlur(
                template_input,
                (blur, blur),
                0,
            )
            roi_input = cv2.GaussianBlur(roi_input, (blur, blur), 0)
            if (
                template_input.shape[0] > roi_input.shape[0]
                or template_input.shape[1] > roi_input.shape[1]
            ):
                return None
            result = cv2.matchTemplate(
                roi_input,
                template_input,
                cv2.TM_SQDIFF_NORMED,
            )
        elif mode == "grayscale":
            result = cv2.matchTemplate(
                cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
                cv2.TM_SQDIFF_NORMED,
                mask=template_mask,
            )
        else:
            result = cv2.matchTemplate(
                image_roi,
                template,
                cv2.TM_SQDIFF_NORMED,
                mask=template_mask,
            )

        locations = np.where(
            result <= self.cfg["monster_detect"]["diff_thres"]
        )
        return [
            ((x, y), result[y, x])
            for y, x in zip(*locations)
        ]

    def _detect_health_bars(self, image_roi, top_left):
        hp_color = np.array(self.cfg["monster_detect"]["hp_bar_color"])
        mask = cv2.inRange(image_roi, hp_color, hp_color)
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        template_height = min(
            image.shape[0]
            for templates in self.monster_templates.values()
            for image, _ in templates
        )
        monsters = []
        for label in range(1, count):
            x, y, _, _, area = stats[label]
            if area < 3:
                continue
            monsters.append(
                {
                    "name": "Health Bar",
                    "position": (top_left[0] + max(0, x), top_left[1] + y + 10),
                    "size": (template_height, 70),
                    "score": 1.0,
                }
            )
        return monsters

    @staticmethod
    def _draw_debug(debug_frame, monsters, top_left, bottom_right):
        draw_rectangle(
            debug_frame,
            top_left,
            (bottom_right[1] - top_left[1], bottom_right[0] - top_left[0]),
            (255, 0, 0),
            "Mob Detection Box",
        )
        for monster in monsters:
            color = (0, 255, 255) if monster["name"] == "Health Bar" else (0, 255, 0)
            draw_rectangle(
                debug_frame,
                monster["position"],
                monster["size"],
                color,
                str(round(monster["score"], 2)),
            )
