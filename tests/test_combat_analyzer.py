import unittest

import numpy as np

from src.engine.CombatAnalyzer import CombatAnalyzer


class CombatAnalyzerTest(unittest.TestCase):
    def test_directional_range_tracks_facing_direction(self):
        cfg = {
            "bot": {"attack": "directional"},
            "directional_attack": {"range_x": 40, "range_y": 20},
        }
        analyzer = CombatAnalyzer(cfg, {})

        self.assertEqual(
            analyzer.attack_range((100, 80), (200, 300, 3), True),
            (60, 70, 100, 90),
        )
        self.assertEqual(
            analyzer.attack_range((100, 80), (200, 300, 3), False),
            (100, 70, 140, 90),
        )

    def test_aoe_range_is_clamped_to_frame(self):
        cfg = {
            "bot": {"attack": "aoe_skill"},
            "aoe_skill": {"range_x": 60, "range_y": 40},
        }
        analyzer = CombatAnalyzer(cfg, {})

        self.assertEqual(
            analyzer.attack_range((10, 5), (100, 120, 3)),
            (0, 0, 40, 25),
        )

    def test_nearest_monster_requires_overlap_and_uses_distance(self):
        cfg = {
            "bot": {"attack": "directional"},
            "directional_attack": {"range_x": 50, "range_y": 40},
            "monster_detect": {"max_mob_area_trigger": 20},
        }
        templates = {"slime": [(np.zeros((5, 6, 3)), None)]}
        analyzer = CombatAnalyzer(cfg, templates)
        far = {"position": (65, 45), "size": (10, 10)}
        near = {"position": (85, 45), "size": (10, 10)}
        outside = {"position": (110, 45), "size": (10, 10)}

        result = analyzer.nearest_monster(
            [far, near, outside],
            player_location=(100, 50),
            frame_shape=(200, 300, 3),
            is_left=True,
        )

        self.assertIs(result, near)

    def test_no_detections_does_not_require_monster_templates(self):
        cfg = {
            "bot": {"attack": "directional"},
            "directional_attack": {"range_x": 50, "range_y": 40},
            "monster_detect": {"max_mob_area_trigger": 20},
        }
        analyzer = CombatAnalyzer(cfg, {})

        self.assertIsNone(
            analyzer.nearest_monster([], (100, 50), (200, 300, 3))
        )


if __name__ == "__main__":
    unittest.main()
