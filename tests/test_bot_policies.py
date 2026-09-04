import unittest

from src.engine.BotPolicies import (
    choose_attack_direction,
    evaluate_other_players,
    evaluate_scheduled_switch,
    evaluate_watchdog,
)


class BotPoliciesTest(unittest.TestCase):
    def test_watchdog_resets_on_movement_and_reports_timeout(self):
        moved = evaluate_watchdog((20, 0), (0, 0), 5.0, 10, 3.0, 7.0)
        self.assertFalse(moved.is_stuck)
        self.assertEqual(moved.watched_location, (20, 0))
        self.assertEqual(moved.checked_at, 7.0)

        stuck = evaluate_watchdog((20, 0), (20, 0), 7.0, 10, 3.0, 11.0)
        self.assertTrue(stuck.is_stuck)
        self.assertEqual(stuck.checked_at, 11.0)

    def test_attack_direction_requires_clear_distance_winner(self):
        left = {"position": (20, 50), "size": (20, 20)}
        close_right = {"position": (110, 50), "size": (20, 20)}
        far_right = {"position": (260, 50), "size": (20, 20)}

        tied = choose_attack_direction(left, close_right, (100, 60))
        self.assertIsNone(tied.direction)

        clear = choose_attack_direction(left, far_right, (100, 60))
        self.assertEqual(clear.direction, "left")

    def test_pixel_channel_policy_keeps_initial_reference_point(self):
        initial = evaluate_other_players([(10, 20)], None, "pixel", 5)
        self.assertFalse(initial.should_change)
        self.assertEqual(initial.center, (10, 20))

        moved = evaluate_other_players([(20, 20)], initial.center, "pixel", 5)
        self.assertTrue(moved.should_change)
        self.assertEqual(moved.center, (10, 20))
        self.assertEqual(moved.movement, 10)

    def test_scheduled_switch_updates_timer_only_when_triggered(self):
        self.assertEqual(
            evaluate_scheduled_switch(True, 10.0, 5.0, 16.0),
            (True, 16.0),
        )
        self.assertEqual(
            evaluate_scheduled_switch(False, 10.0, 5.0, 16.0),
            (False, 10.0),
        )


if __name__ == "__main__":
    unittest.main()
