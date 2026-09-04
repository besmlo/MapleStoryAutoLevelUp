import unittest

import numpy as np

from src.engine.MapScanner import MapScanner


class MapScannerTest(unittest.TestCase):
    def test_initialize_adds_padding_and_removes_route_colors(self):
        scanner = MapScanner(map_padding=2, route_colors=[(255, 0, 0)])
        minimap = np.full((3, 4, 3), 7, dtype=np.uint8)
        minimap[1, 1] = (0, 0, 255)

        image, route = scanner.initialize(minimap)

        self.assertEqual(image.shape, (7, 8, 3))
        self.assertTrue(np.array_equal(image[3, 3], (0, 0, 255)))
        self.assertTrue(np.array_equal(route[3, 3], (0, 0, 0)))

    def test_capacity_expansion_keeps_map_route_and_origin_aligned(self):
        scanner = MapScanner(map_padding=2, route_colors=[])
        image = np.ones((5, 6, 3), dtype=np.uint8)
        route = np.full((5, 6, 3), 2, dtype=np.uint8)

        expanded = scanner.ensure_capacity(
            image,
            route,
            origin=(0, 0),
            region_shape=(4, 4),
        )

        self.assertEqual(expanded.origin, (2, 2))
        self.assertEqual(expanded.image.shape, expanded.route.shape)
        self.assertTrue(np.array_equal(expanded.image[2:7, 2:8], image))
        self.assertTrue(np.array_equal(expanded.route[2:7, 2:8], route))

    def test_merge_only_fills_unseen_pixels(self):
        scanner = MapScanner(map_padding=0, route_colors=[])
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[1, 2] = (9, 9, 9)
        minimap = np.full((2, 3, 3), 4, dtype=np.uint8)

        scanner.merge_unseen(image, minimap, (1, 1))

        self.assertTrue(np.array_equal(image[1, 2], (9, 9, 9)))
        self.assertTrue(np.array_equal(image[2, 3], (4, 4, 4)))


if __name__ == "__main__":
    unittest.main()
