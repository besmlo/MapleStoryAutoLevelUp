"""Pure combat geometry used by the AutoBot decision layer."""


class CombatAnalyzer:
    def __init__(self, cfg, monster_templates):
        self.cfg = cfg
        self.monster_templates = monster_templates

    def attack_range(self, player_location, frame_shape, is_left=True):
        player_x, player_y = player_location
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2
            dy = self.cfg["aoe_skill"]["range_y"] // 2
            return (
                max(0, player_x - dx),
                max(0, player_y - dy),
                min(frame_shape[1], player_x + dx),
                min(frame_shape[0], player_y + dy),
            )

        if self.cfg["bot"]["attack"] == "directional":
            range_x = self.cfg["directional_attack"]["range_x"]
            range_y = self.cfg["directional_attack"]["range_y"]
            x0 = player_x - range_x if is_left else player_x
            x1 = player_x if is_left else player_x + range_x
            y0 = player_y - range_y // 2
            return x0, y0, x1, y0 + range_y

        raise RuntimeError(
            f"Unsupported attack mode: {self.cfg['bot']['attack']}"
        )

    def nearest_monster(
        self,
        monsters,
        player_location,
        frame_shape,
        is_left=True,
    ):
        if not monsters:
            return None
        x0, y0, x1, y1 = self.attack_range(
            player_location,
            frame_shape,
            is_left,
        )
        min_template_area = min(
            image.shape[0] * image.shape[1]
            for images in self.monster_templates.values()
            for image, _ in images
        )
        area_threshold = min(
            min_template_area,
            self.cfg["monster_detect"]["max_mob_area_trigger"],
        )

        nearest = None
        nearest_distance = float("inf")
        player_x, player_y = player_location
        for monster in monsters:
            monster_x, monster_y = monster["position"]
            monster_width, monster_height = monster["size"]
            intersection_width = max(
                0,
                min(x1, monster_x + monster_width) - max(x0, monster_x),
            )
            intersection_height = max(
                0,
                min(y1, monster_y + monster_height) - max(y0, monster_y),
            )
            if intersection_width * intersection_height < area_threshold:
                continue

            center = (
                monster_x + monster_width // 2,
                monster_y + monster_height // 2,
            )
            distance = abs(center[0] - player_x) + abs(center[1] - player_y)
            if distance < nearest_distance:
                nearest = monster
                nearest_distance = distance
        return nearest
