import unittest

import numpy as np

from src.engine.RouteAnalyzer import find_nearest_actions


class RouteAnalyzerTest(unittest.TestCase):
    def test_finds_regular_and_vertical_actions_independently(self):
        route = np.zeros((9, 9, 3), dtype=np.uint8)
        route[4, 2] = (1, 2, 3)
        route[2, 4] = (4, 5, 6)

        regular, vertical, bounds = find_nearest_actions(
            route,
            player_location=(4, 4),
            search_range=4,
            color_code={(1, 2, 3): "left none none"},
            color_code_up_down={(4, 5, 6): "none up none"},
        )

        self.assertEqual(regular["pixel"], (2, 4))
        self.assertEqual(vertical["pixel"], (4, 2))
        self.assertEqual(bounds, (0, 0, 8, 8))

    def test_search_bounds_are_clamped_at_route_edge(self):
        route = np.zeros((5, 6, 3), dtype=np.uint8)

        regular, vertical, bounds = find_nearest_actions(
            route,
            player_location=(1, 1),
            search_range=3,
            color_code={},
            color_code_up_down={},
        )

        self.assertIsNone(regular)
        self.assertIsNone(vertical)
        self.assertEqual(bounds, (0, 0, 4, 4))


if __name__ == "__main__":
    unittest.main()
