from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.engine.MapBundleValidator import (
    MapBundleValidator,
    register_map_bundle,
    validate_map_id,
)
from src.utils.common import load_yaml


class MapBundleTest(unittest.TestCase):
    def setUp(self):
        self.validator = MapBundleValidator(
            {"255,0,0": {"command": "none none goal"}}
        )

    def test_valid_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            map_dir = Path(directory)
            image = np.zeros((20, 30, 3), dtype=np.uint8)
            route = image.copy()
            route[10, 10] = (0, 0, 255)
            cv2.imwrite(str(map_dir / "map.png"), image)
            cv2.imwrite(str(map_dir / "route1.png"), route)

            result = self.validator.validate(map_dir)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.route_count, 1)

    def test_bundle_rejects_mismatched_route_and_missing_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            map_dir = Path(directory)
            cv2.imwrite(
                str(map_dir / "map.png"),
                np.zeros((20, 30, 3), dtype=np.uint8),
            )
            cv2.imwrite(
                str(map_dir / "route1.png"),
                np.zeros((10, 30, 3), dtype=np.uint8),
            )

            result = self.validator.validate(map_dir)

        self.assertFalse(result.is_valid)
        self.assertIn("does not match", result.errors[0])

    def test_register_map_bundle_preserves_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "config_data.yaml"
            data_path.write_text(
                "eng_to_cn:\n  old_map: 舊地圖\n"
                "map_mobs_mapping:\n  old_map: [old_mob]\n",
                encoding="utf-8",
            )

            register_map_bundle(
                "new_map",
                "新地圖",
                ["green_mushroom"],
                data_path=data_path,
            )
            data = load_yaml(data_path)

        self.assertEqual(data["eng_to_cn"]["old_map"], "舊地圖")
        self.assertEqual(data["eng_to_cn"]["new_map"], "新地圖")
        self.assertEqual(
            data["map_mobs_mapping"]["new_map"], ("green_mushroom",)
        )

    def test_map_id_rejects_paths(self):
        with self.assertRaises(ValueError):
            validate_map_id("../existing-map")


if __name__ == "__main__":
    unittest.main()
