"""Route-map lookup helpers with no UI or keyboard dependencies."""

import numpy as np


def find_nearest_actions(
    route_image,
    player_location,
    search_range,
    color_code,
    color_code_up_down,
):
    player_x, player_y = player_location
    height, width = route_image.shape[:2]
    bounds = (
        max(0, player_x - search_range),
        max(0, player_y - search_range),
        min(width, player_x + search_range),
        min(height, player_y + search_range),
    )
    x_min, y_min, x_max, y_max = bounds
    nearest = None
    nearest_up_down = None
    min_distance = float("inf")
    min_up_down_distance = float("inf")

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            pixel = tuple(route_image[y, x])
            distance = abs(x - player_x) + abs(y - player_y)
            if pixel in color_code and distance < min_distance:
                nearest = {
                    "pixel": (x, y),
                    "color": pixel,
                    "command": color_code[pixel],
                    "distance": distance,
                }
                min_distance = distance
            if (
                pixel in color_code_up_down
                and distance < min_up_down_distance
            ):
                nearest_up_down = {
                    "pixel": (x, y),
                    "color": pixel,
                    "command": color_code_up_down[pixel],
                    "distance": distance,
                }
                min_up_down_distance = distance
    return nearest, nearest_up_down, bounds


def merge_route_commands(nearest, nearest_up_down, is_on_ladder):
    if nearest and nearest_up_down:
        if nearest["distance"] < nearest_up_down["distance"]:
            move_x, move_y, action = nearest["command"].split()
            _, vertical, _ = nearest_up_down["command"].split()
            if move_y == "none" and is_on_ladder:
                move_y = vertical
        else:
            move_x, move_y, action = nearest_up_down["command"].split()
            horizontal, _, _ = nearest["command"].split()
            if move_x == "none" and is_on_ladder:
                move_x = horizontal
        return move_x, move_y, action
    if nearest:
        return tuple(nearest["command"].split())
    if nearest_up_down:
        return tuple(nearest_up_down["command"].split())
    return None


def find_edge_side(
    route_image,
    player_location,
    trigger_width,
    trigger_height,
    color_code,
):
    player_x, player_y = player_location
    height, width = route_image.shape[:2]
    x_min = max(0, player_x - trigger_width // 2)
    x_max = min(width, player_x + trigger_width // 2)
    y_min = max(0, player_y - trigger_height // 2)
    y_max = min(height, player_y + trigger_height // 2)
    image_roi = route_image[y_min:y_max, x_min:x_max]
    coordinates = np.column_stack(
        np.where(np.all(image_roi == color_code, axis=2))
    )
    if coordinates.size == 0:
        return ""

    mean_x = np.mean(coordinates[:, 1])
    return "edge on left" if mean_x < player_x else "edge on right"
