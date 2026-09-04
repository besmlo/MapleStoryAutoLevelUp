import unittest

import numpy as np

from src.engine.MonsterDetector import MonsterDetector


def make_config(mode="template_free", bot_mode="normal", hp_bar=False):
    return {
        "bot": {"mode": bot_mode},
        "character": {"width": 10, "height": 10},
        "monster_detect": {
            "mode": mode,
            "diff_thres": 0.01,
            "contour_blur": 3,
            "with_enemy_hp_bar": hp_bar,
            "hp_bar_color": [20, 200, 20],
        },
    }


class MonsterDetectorTest(unittest.TestCase):
    def test_template_free_detection_ignores_player_region(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        frame[10:50, 10:50] = 0
        debug = frame.copy()
        templates = {"slime": [(np.zeros((5, 5, 3)), None)]}
        detector = MonsterDetector(make_config(), templates)

        monsters = detector.detect(
            frame,
            debug,
            player_location=(80, 80),
            top_left=(0, 0),
            bottom_right=(100, 100),
        )

        self.assertEqual(len(monsters), 1)
        x, y = monsters[0]["position"]
        height, width = monsters[0]["size"]
        self.assertTrue(9 <= x <= 12)
        self.assertTrue(9 <= y <= 12)
        self.assertGreater(height * width, 1000)

    def test_template_free_mode_preserves_empty_template_behavior(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        detector = MonsterDetector(make_config(), {})

        monsters = detector.detect(
            frame,
            frame.copy(),
            player_location=(25, 25),
            top_left=(0, 0),
            bottom_right=(50, 50),
        )

        self.assertEqual(monsters, [])

    def test_patrol_skips_templates_but_can_detect_health_bar(self):
        hp_color = (20, 200, 20)
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[5, 7:10] = hp_color
        debug = frame.copy()
        templates = {"slime": [(np.zeros((12, 8, 3)), None)]}
        detector = MonsterDetector(
            make_config(bot_mode="patrol", hp_bar=True),
            templates,
        )

        monsters = detector.detect(
            frame,
            debug,
            player_location=(40, 40),
            top_left=(0, 0),
            bottom_right=(80, 60),
        )

        self.assertEqual(len(monsters), 1)
        self.assertEqual(monsters[0]["name"], "Health Bar")
        self.assertEqual(monsters[0]["position"], (7, 15))
        self.assertEqual(monsters[0]["size"], (12, 70))

    def test_unknown_detection_mode_returns_no_matches(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        detector = MonsterDetector(make_config(mode="unknown"), {})

        result = detector.detect(
            frame,
            frame.copy(),
            player_location=(10, 10),
            top_left=(0, 0),
            bottom_right=(20, 20),
        )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
