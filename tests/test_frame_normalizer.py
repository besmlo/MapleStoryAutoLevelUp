import unittest

import numpy as np

from src.engine.FrameNormalizer import normalize_game_frame


class FrameNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "game_window": {
                "size": (752, 1282),
                "ratio_tolerance": 0.08,
            }
        }

    def test_missing_frame_returns_diagnostic(self):
        result = normalize_game_frame(None, self.cfg)
        self.assertIsNone(result.image)
        self.assertIn("Failed to capture", result.error)

    def test_supported_aspect_ratio_is_normalized(self):
        frame = np.zeros((932, 1602, 3), dtype=np.uint8)
        result = normalize_game_frame(frame, self.cfg)
        self.assertIsNone(result.error)
        self.assertTrue(result.was_resized)
        self.assertEqual(result.image.shape, (759, 1296, 3))

    def test_invalid_aspect_ratio_is_rejected(self):
        frame = np.zeros((800, 800, 3), dtype=np.uint8)
        result = normalize_game_frame(frame, self.cfg)
        self.assertIsNone(result.image)
        self.assertIn("aspect ratio", result.error)

    def test_static_test_image_can_skip_aspect_check(self):
        frame = np.zeros((800, 800, 3), dtype=np.uint8)
        result = normalize_game_frame(
            frame,
            self.cfg,
            skip_aspect_check=True,
        )
        self.assertEqual(result.image.shape, (759, 1296, 3))


if __name__ == "__main__":
    unittest.main()
