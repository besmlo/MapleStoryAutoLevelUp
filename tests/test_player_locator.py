import unittest
from unittest.mock import patch

import numpy as np

from src.engine.PlayerLocator import PlayerLocator


class PlayerLocatorTest(unittest.TestCase):
    def test_global_map_location_applies_configured_offset(self):
        cfg = {"minimap": {"offset": [3, -2]}}
        locator = PlayerLocator(cfg)
        map_image = np.zeros((30, 40, 3), dtype=np.uint8)
        minimap = np.zeros((5, 6, 3), dtype=np.uint8)
        debug = map_image.copy()

        with patch(
            "src.engine.PlayerLocator.find_pattern_sqdiff",
            return_value=((10, 8), 0.01, False),
        ):
            result = locator.on_global_map(
                map_image,
                minimap,
                player_on_minimap=(4, 3),
                route_debug=debug,
            )

        self.assertEqual(result.minimap_origin, (10, 8))
        self.assertEqual(result.player, (17, 9))

    def test_party_red_bar_returns_player_and_bar_locations(self):
        cfg = {
            "camera": {"y_start": 0, "y_end": 50},
            "party_red_bar": {
                "lower_red": [0, 80, 80],
                "upper_red": [5, 100, 100],
                "offset": [2, -3],
            },
        }
        frame = np.zeros((50, 80, 3), dtype=np.uint8)
        frame[12:18, 10:30] = (0, 0, 255)

        player, red_bar = PlayerLocator(cfg).by_party_red_bar(
            frame,
            frame.copy(),
            minimap_origin=(60, 30),
            minimap_shape=(5, 5),
        )

        self.assertEqual(red_bar, (10, 12))
        self.assertEqual(player, (12, 9))

    def test_unknown_nametag_mode_returns_none(self):
        cfg = {
            "camera": {"y_start": 0, "y_end": 10},
            "nametag": {"mode": "unknown"},
        }
        image = np.zeros((10, 10), dtype=np.uint8)

        result = PlayerLocator(cfg).by_nametag(
            image,
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            previous_location=(0, 0),
            is_first_frame=True,
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
