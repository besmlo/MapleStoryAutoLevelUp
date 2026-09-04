"""Minimap stitching operations used by the route recorder."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.common import find_pattern_sqdiff, to_opencv_hsv


@dataclass(frozen=True)
class ExpandedMap:
    image: np.ndarray
    route: np.ndarray | None
    origin: tuple[int, int]


class MapScanner:
    """Build and extend a global map from the visible in-game minimap."""

    def __init__(self, map_padding, route_colors):
        self.map_padding = map_padding
        self.route_colors = tuple(route_colors)

    def initialize(self, minimap, existing_map=None):
        image = minimap.copy() if existing_map is None else existing_map.copy()
        if existing_map is None:
            image = cv2.copyMakeBorder(
                image,
                top=self.map_padding,
                bottom=self.map_padding,
                left=self.map_padding,
                right=self.map_padding,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        image = self.replace_hsv_components(image, (55, 40, 80), (60, 100, 100))
        image = self.replace_hsv_components(image, (0, 80, 80), (5, 100, 100))
        return image, self.remove_route_colors(image.copy())

    def locate(self, image, minimap):
        mask = np.any(minimap != [0, 0, 0], axis=2).astype(np.uint8) * 255
        origin, score, _ = find_pattern_sqdiff(image, minimap, mask=mask)
        return origin, score

    def ensure_capacity(self, image, route, origin, region_shape):
        x, y = origin
        h, w = region_shape[:2]
        map_h, map_w = image.shape[:2]
        pad = self.map_padding

        expand_top = max(0, pad - y) if y < pad else 0
        expand_left = max(0, pad - x) if x < pad else 0
        expand_bottom = max(0, y + h + pad - map_h)
        expand_right = max(0, x + w + pad - map_w)
        if not any((expand_top, expand_bottom, expand_left, expand_right)):
            return ExpandedMap(image, route, origin)

        expanded_image = cv2.copyMakeBorder(
            image,
            top=expand_top,
            bottom=expand_bottom,
            left=expand_left,
            right=expand_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        expanded_route = route
        if route is not None:
            expanded_route = cv2.copyMakeBorder(
                route,
                top=expand_top,
                bottom=expand_bottom,
                left=expand_left,
                right=expand_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        return ExpandedMap(
            expanded_image,
            expanded_route,
            (x + expand_left, y + expand_top),
        )

    @staticmethod
    def merge_unseen(image, minimap, origin):
        x, y = origin
        h, w = minimap.shape[:2]
        map_slice = image[y:y + h, x:x + w]
        black_mask = np.all(map_slice == [0, 0, 0], axis=2)
        map_slice[black_mask] = minimap[black_mask]
        return image

    def remove_route_colors(self, image):
        for rgb in self.route_colors:
            image[np.all(image == rgb[::-1], axis=2)] = (0, 0, 0)
        return image

    @staticmethod
    def replace_hsv_components(
        image,
        lower_hsv,
        upper_hsv,
        replace_color=(0, 0, 0),
        minimum_area=10,
    ):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv_image,
            to_opencv_hsv(lower_hsv),
            to_opencv_hsv(upper_hsv),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] > minimum_area:
                image[labels == label] = replace_color
        return image
