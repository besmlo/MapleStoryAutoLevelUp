from dataclasses import dataclass, field
from pathlib import Path
import os
import re

import cv2
import numpy as np
from ruamel.yaml import YAML


MAP_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_map_id(map_id):
    if not MAP_ID_PATTERN.fullmatch(map_id):
        raise ValueError(
            "Map ID may contain only letters, numbers, underscores, and hyphens"
        )
    return map_id


@dataclass
class MapBundleValidation:
    errors: list[str] = field(default_factory=list)
    route_count: int = 0

    @property
    def is_valid(self):
        return not self.errors


class MapBundleValidator:
    def __init__(self, route_color_code):
        self.goal_colors_bgr = []
        for color, metadata in route_color_code.items():
            if metadata.get("command", "").split()[-1:] == ["goal"]:
                rgb = tuple(map(int, color.split(",")))
                self.goal_colors_bgr.append(rgb[::-1])

    def validate(self, map_dir):
        map_dir = Path(map_dir)
        result = MapBundleValidation()
        map_path = map_dir / "map.png"
        map_image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if map_image is None:
            result.errors.append("map.png is missing or unreadable")
            return result

        route_paths = sorted(
            path
            for path in map_dir.glob("route*.png")
            if path.name != "route_rest.png"
        )
        result.route_count = len(route_paths)
        if not route_paths:
            result.errors.append("At least one route image is required")
            return result

        for route_path in route_paths:
            route = cv2.imread(str(route_path), cv2.IMREAD_COLOR)
            if route is None:
                result.errors.append(f"{route_path.name} is unreadable")
                continue
            if route.shape[:2] != map_image.shape[:2]:
                result.errors.append(
                    f"{route_path.name} size {route.shape[:2]} does not match "
                    f"map.png {map_image.shape[:2]}"
                )
                continue
            has_goal = any(
                np.any(np.all(route == color, axis=2))
                for color in self.goal_colors_bgr
            )
            if not has_goal:
                result.errors.append(f"{route_path.name} has no goal marker")

        return result


def register_map_bundle(
    map_id,
    display_name,
    monsters,
    data_path="config/config_data.yaml",
):
    validate_map_id(map_id)
    if not monsters:
        raise ValueError("Select at least one monster")

    yaml = YAML()
    yaml.preserve_quotes = True
    data_path = Path(data_path)
    with data_path.open("r", encoding="utf-8") as source:
        data = yaml.load(source)

    data.setdefault("eng_to_cn", {})[map_id] = display_name or map_id
    data.setdefault("map_mobs_mapping", {})[map_id] = list(monsters)

    temp_path = data_path.with_suffix(data_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as destination:
        yaml.dump(data, destination)
    os.replace(temp_path, data_path)
