import unittest

import numpy as np

from src.engine.RouteAnalyzer import (
    find_edge_side,
    find_nearest_actions,
    merge_route_commands,
)


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

    def test_ladder_command_combines_horizontal_and_vertical_routes(self):
        horizontal = {"command": "left none none", "distance": 2}
        vertical = {"command": "none up none", "distance": 3}

        self.assertEqual(
            merge_route_commands(horizontal, vertical, is_on_ladder=True),
            ("left", "up", "none"),
        )
        self.assertEqual(
            merge_route_commands(horizontal, vertical, is_on_ladder=False),
            ("left", "none", "none"),
        )

    def test_edge_side_preserves_existing_route_coordinate_rule(self):
        route = np.zeros((20, 30, 3), dtype=np.uint8)
        color = (1, 2, 3)
        route[10, 7] = color

        self.assertEqual(
            find_edge_side(route, (10, 10), 10, 10, color),
            "edge on left",
        )
        self.assertEqual(
            find_edge_side(route, (10, 10), 10, 10, (9, 9, 9)),
            "",
        )


if __name__ == "__main__":
    unittest.main()
