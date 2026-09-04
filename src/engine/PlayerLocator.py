"""Player-location detectors shared by the AutoBot orchestration layer."""

from dataclasses import dataclass

import cv2

from src.utils.common import (
    draw_rectangle,
    find_pattern_sqdiff,
    get_mask,
    to_opencv_hsv,
)
from src.utils.logger import logger


@dataclass(frozen=True)
class NameTagLocation:
    player: tuple[int, int]
    nametag: tuple[int, int]


@dataclass(frozen=True)
class GlobalMapLocation:
    player: tuple[int, int]
    minimap_origin: tuple[int, int]


class PlayerLocator:
    def __init__(self, cfg):
        self.cfg = cfg

    def by_nametag(
        self,
        frame_gray,
        debug_frame,
        nametag_image,
        nametag_gray,
        previous_location,
        is_first_frame,
    ):
        camera_start = self.cfg["camera"]["y_start"]
        camera = frame_gray[camera_start:self.cfg["camera"]["y_end"], :]
        mode = self.cfg["nametag"]["mode"]
        if mode == "white_mask":
            camera = cv2.GaussianBlur(camera, (3, 3), 0)
            template = cv2.GaussianBlur(nametag_gray, (3, 3), 0)
            image_roi = cv2.inRange(camera, 150, 255)
            template = cv2.inRange(template, 150, 255)
        elif mode == "grayscale":
            image_roi = camera
            template = nametag_gray
        elif mode == "histogram_eq":
            template_equalized = cv2.equalizeHist(nametag_gray)
            camera_equalized = cv2.equalizeHist(camera)
            _, template = cv2.threshold(
                template_equalized,
                150,
                255,
                cv2.THRESH_BINARY,
            )
            _, image_roi = cv2.threshold(
                camera_equalized,
                150,
                255,
                cv2.THRESH_BINARY,
            )
        else:
            logger.error(f"Unsupported nametag detection mode: {mode}")
            return None

        pad_y, pad_x = nametag_image.shape[:2]
        image_roi = cv2.copyMakeBorder(
            image_roi,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            borderType=cv2.BORDER_REPLICATE,
        )
        last_result = None
        if not is_first_frame:
            last_result = (
                previous_location[0] + pad_x,
                previous_location[1] + pad_y - camera_start,
            )

        _, width = template.shape
        split_count = max(1, width // self.cfg["nametag"]["split_width"])
        split_width = width // split_count
        background_mask = get_mask(nametag_image, (0, 255, 0))
        matches = []
        for index in range(split_count):
            x_start = index * split_width
            x_end = (index + 1) * split_width if index < split_count - 1 else width
            cached_location = (
                (last_result[0] + x_start, last_result[1])
                if last_result
                else None
            )
            split = template[:, x_start:x_end]
            location, score, is_cached = find_pattern_sqdiff(
                image_roi,
                split,
                last_result=cached_location,
                mask=background_mask[:, x_start:x_end],
                global_threshold=self.cfg["nametag"]["global_diff_thres"],
            )
            matches.append(
                (
                    f"{index + 1}/{split_count}",
                    location,
                    score,
                    is_cached,
                    x_start,
                )
            )

        matches.sort(key=lambda match: (not match[3], match[2]))
        tag_type, location, score, is_cached, offset_x = matches[0]
        detected_location = (
            location[0] - offset_x - pad_x,
            location[1] - pad_y + camera_start,
        )
        nametag_location = previous_location
        if score < self.cfg["nametag"]["diff_thres"]:
            nametag_location = detected_location

        player_location = (
            nametag_location[0] + width // 2,
            nametag_location[1] - self.cfg["nametag"]["offset"][1],
        )
        draw_rectangle(
            debug_frame,
            nametag_location,
            nametag_image.shape,
            (0, 255, 0),
            "",
        )
        text = (
            f"NameTag,{round(score, 2)},"
            f"{'cached' if is_cached else 'missed'},{tag_type}"
        )
        cv2.putText(
            debug_frame,
            text,
            (
                nametag_location[0],
                nametag_location[1] + nametag_image.shape[0] + 30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        return NameTagLocation(player_location, nametag_location)

    def by_party_red_bar(
        self,
        frame,
        debug_frame,
        minimap_origin,
        minimap_shape,
    ):
        image = frame.copy()
        minimap_x, minimap_y = minimap_origin
        minimap_h, minimap_w = minimap_shape[:2]
        image[
            minimap_y:minimap_y + minimap_h,
            minimap_x:minimap_x + minimap_w,
        ] = 0
        camera_start = self.cfg["camera"]["y_start"]
        camera = image[camera_start:self.cfg["camera"]["y_end"], :]
        image_hsv = cv2.cvtColor(camera, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            image_hsv,
            to_opencv_hsv(self.cfg["party_red_bar"]["lower_red"]),
            to_opencv_hsv(self.cfg["party_red_bar"]["upper_red"]),
        )
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        boxes = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            fill_rate = float(area) / (height * width)
            if (
                5 <= height <= 7
                and 1 <= width <= 50
                and area >= 10
                and fill_rate >= 0.7
            ):
                boxes.append((x, y, width, height))
        if not boxes:
            return None, None

        x, y, width, height = max(boxes, key=lambda box: box[2] * box[3])
        red_bar = (x, y + camera_start)
        player = (
            red_bar[0] + self.cfg["party_red_bar"]["offset"][0],
            red_bar[1] + self.cfg["party_red_bar"]["offset"][1],
        )
        draw_rectangle(
            debug_frame,
            red_bar,
            (height, width),
            (0, 255, 0),
            "party red bar",
            thickness=1,
            text_height=0.4,
        )
        return player, red_bar

    def on_global_map(
        self,
        map_image,
        minimap,
        player_on_minimap,
        route_debug,
    ):
        origin, score, _ = find_pattern_sqdiff(map_image, minimap)
        offset_x, offset_y = self.cfg["minimap"]["offset"]
        player = (
            origin[0] + player_on_minimap[0] + offset_x,
            origin[1] + player_on_minimap[1] + offset_y,
        )
        bottom_right = (
            origin[0] + minimap.shape[1],
            origin[1] + minimap.shape[0],
        )
        cv2.rectangle(route_debug, origin, bottom_right, (0, 255, 255), 1)
        cv2.putText(
            route_debug,
            f"Minimap,score({round(score, 2)})",
            (origin[0], origin[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )
        cv2.circle(
            route_debug,
            player,
            radius=2,
            color=(0, 255, 255),
            thickness=-1,
        )
        return GlobalMapLocation(player, origin)
