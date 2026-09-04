"""Route-map lookup helpers with no UI or keyboard dependencies."""


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
