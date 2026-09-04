"""Configuration parsing and image-resource loading for the AutoBot engine."""

import glob
from dataclasses import dataclass, field

import cv2

from src.utils.common import (
    get_mask,
    load_image,
    mask_route_colors,
    normalize_language_code,
)


class ConfigResourceError(ValueError):
    """Raised when a selected bot configuration cannot be loaded."""


@dataclass
class BotResources:
    color_code: dict = field(default_factory=dict)
    color_code_up_down: dict = field(default_factory=dict)
    img_map: object = None
    img_routes: list = field(default_factory=list)
    monsters_info: dict = field(default_factory=dict)
    img_nametag: object = None
    img_nametag_gray: object = None
    img_create_party_enable: object = None
    img_create_party_disable: object = None
    img_login_button: object = None


def parse_color_codes(encoded_colors):
    return {
        tuple(map(int, color.split(","))): metadata
        for color, metadata in encoded_colors.items()
    }


def load_bot_resources(cfg, data):
    resources = BotResources(
        color_code=parse_color_codes(cfg["route"]["color_code"]),
        color_code_up_down=parse_color_codes(
            cfg["route"]["color_code_up_down"]
        ),
    )

    if cfg["bot"]["mode"] == "normal":
        _load_normal_mode_resources(resources, cfg, data)

    if cfg["nametag"]["enable"]:
        name = cfg["nametag"]["name"]
        resources.img_nametag = load_image(f"nametag/{name}.png")
        resources.img_nametag_gray = load_image(
            f"nametag/{name}.png",
            cv2.IMREAD_GRAYSCALE,
        )

    language = normalize_language_code(cfg["system"]["language"])
    cfg["system"]["language"] = language
    resources.img_create_party_enable = load_image(
        f"misc/party_button_create_enable_{language}.png"
    )
    resources.img_create_party_disable = load_image(
        f"misc/party_button_create_disable_{language}.png"
    )
    resources.img_login_button = load_image(
        f"misc/login_button_{language}.png"
    )
    return resources


def _load_normal_mode_resources(resources, cfg, data):
    map_name = cfg["bot"]["map"]
    if map_name not in data["map_mobs_mapping"]:
        raise ConfigResourceError(
            f"Invalid map name: {map_name}. Not supported in "
            "config/config_data.yaml."
        )

    resources.img_map = load_image(
        f"minimaps/{map_name}/map.png",
        cv2.IMREAD_COLOR,
    )
    route_files = sorted(glob.glob(f"minimaps/{map_name}/route*.png"))
    route_files = [
        path for path in route_files if not path.endswith("route_rest.png")
    ]
    for route_file in route_files:
        route = cv2.cvtColor(load_image(route_file), cv2.COLOR_BGR2RGB)
        route = mask_route_colors(
            resources.img_map,
            route,
            cfg["route"]["color_code"],
        )
        route = mask_route_colors(
            resources.img_map,
            route,
            cfg["route"]["color_code_up_down"],
        )
        resources.img_routes.append(route)

    for monster_name in data["map_mobs_mapping"][map_name]:
        images = []
        pattern = f"monster/{monster_name}/{monster_name}*.png"
        for image_path in glob.glob(pattern):
            image = load_image(image_path)
            images.append((image, get_mask(image, (0, 255, 0))))
            flipped = cv2.flip(image, 1)
            images.append((flipped, get_mask(flipped, (0, 255, 0))))
        if not images:
            raise ConfigResourceError(
                f"No images found in monster/{monster_name}/{monster_name}*"
            )
        resources.monsters_info[monster_name] = images
