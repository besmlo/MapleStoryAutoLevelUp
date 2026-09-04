import unittest
from unittest.mock import patch

from src.engine.BotConfigLoader import (
    ConfigResourceError,
    load_bot_resources,
    parse_color_codes,
)


class BotConfigLoaderTest(unittest.TestCase):
    def test_parse_color_codes_uses_rgb_tuple_keys(self):
        parsed = parse_color_codes({"1,2,3": {"command": "left none none"}})
        self.assertEqual(parsed[(1, 2, 3)]["command"], "left none none")

    def test_aux_mode_normalizes_language_and_loads_shared_assets(self):
        cfg = {
            "bot": {"mode": "aux"},
            "route": {"color_code": {}, "color_code_up_down": {}},
            "nametag": {"enable": False, "name": "unused"},
            "system": {"language": "english"},
        }

        with patch(
            "src.engine.BotConfigLoader.load_image",
            side_effect=lambda path, *args: path,
        ):
            resources = load_bot_resources(cfg, {"map_mobs_mapping": {}})

        self.assertEqual(cfg["system"]["language"], "eng")
        self.assertEqual(
            resources.img_login_button,
            "misc/login_button_eng.png",
        )
        self.assertEqual(resources.img_routes, [])

    def test_normal_mode_rejects_unknown_map_before_loading_images(self):
        cfg = {
            "bot": {"mode": "normal", "map": "missing"},
            "route": {"color_code": {}, "color_code_up_down": {}},
            "nametag": {"enable": False, "name": "unused"},
            "system": {"language": "eng"},
        }

        with self.assertRaisesRegex(ConfigResourceError, "Invalid map name"):
            load_bot_resources(cfg, {"map_mobs_mapping": {}})


if __name__ == "__main__":
    unittest.main()
