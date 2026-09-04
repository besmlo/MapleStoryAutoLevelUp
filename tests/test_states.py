import time
import unittest

import numpy as np

from src.engine.FiniteStateMachine import FiniteStateMachine
from src.states.auxiliary import AuxiliaryState
from src.states.base_state import State
from src.states.hunting import HuntingState
from src.states.patrol import PatrolState


class FakeKeyboard:
    def __init__(self):
        self.commands = []

    def set_command(self, command):
        self.commands.append(command)


class FakeBot:
    def __init__(self, stuck=False):
        self.kb = FakeKeyboard()
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        self.stuck = stuck
        self.calls = []
        self.loc_player = (5, 5)
        self.img_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.t_last_attack = time.time()
        self.cfg = {
            "patrol": {
                "range": (0.2, 0.8),
                "turn_point_thres": 0,
                "patrol_attack_interval": 60,
            }
        }

    def update_cmd_by_route(self):
        self.calls.append("route")
        self.cmd_move_x = "left"

    def check_reach_goal(self):
        self.calls.append("goal")

    def update_cmd_by_mob_detection(self):
        self.calls.append("mob")
        self.cmd_action = "attack"

    def is_player_stuck(self):
        self.calls.append("stuck")
        return self.stuck

    def update_cmd_by_random(self):
        self.calls.append("recover")
        self.cmd_move_x = "right"
        self.cmd_move_y = "down"
        self.cmd_action = "jump"


class StateBehaviorTest(unittest.TestCase):
    def test_base_state_requires_per_frame_behavior(self):
        with self.assertRaises(NotImplementedError):
            State("base", FakeBot()).on_frame()

    def test_hunting_preserves_behavior_order_and_command(self):
        bot = FakeBot(stuck=False)

        HuntingState("hunting", bot).on_frame()

        self.assertEqual(bot.calls, ["route", "goal", "mob", "stuck"])
        self.assertEqual(bot.kb.commands, ["left none attack"])

    def test_hunting_recovery_overrides_command_when_stuck(self):
        bot = FakeBot(stuck=True)

        HuntingState("hunting", bot).on_frame()

        self.assertEqual(bot.kb.commands, ["right down jump"])
        self.assertIn("recover", bot.calls)

    def test_patrol_turns_at_boundary_and_sends_command(self):
        bot = FakeBot(stuck=False)

        PatrolState("patrol", bot).on_frame()

        self.assertEqual(bot.cmd_move_x, "right")
        self.assertEqual(bot.kb.commands, ["right none attack"])

    def test_auxiliary_mode_does_not_send_commands(self):
        bot = FakeBot()

        AuxiliaryState("aux", bot).on_frame()

        self.assertEqual(bot.kb.commands, [])


class TrackingState(State):
    def __init__(self, name, events):
        super().__init__(name, bot=None)
        self.events = events

    def on_enter(self):
        self.events.append(f"enter:{self.name}")

    def on_exit(self):
        self.events.append(f"exit:{self.name}")

    def on_frame(self):
        self.events.append(f"frame:{self.name}")


class StateMachineTest(unittest.TestCase):
    def test_state_selection_runs_lifecycle_and_frame(self):
        events = []
        machine = FiniteStateMachine()
        machine.add_state(TrackingState("first", events))
        machine.add_state(TrackingState("second", events))

        self.assertTrue(machine.set_state("first"))
        machine.run_frame()
        self.assertTrue(machine.set_state("second"))

        self.assertEqual(
            events,
            ["enter:first", "frame:first", "exit:first", "enter:second"],
        )

    def test_unknown_state_is_rejected_without_losing_current_state(self):
        events = []
        machine = FiniteStateMachine()
        machine.add_state(TrackingState("known", events))
        machine.set_state("known")

        self.assertFalse(machine.set_state("missing"))
        self.assertEqual(machine.state.name, "known")

    def test_frame_requires_selected_state(self):
        with self.assertRaisesRegex(RuntimeError, "not been initialized"):
            FiniteStateMachine().run_frame()


if __name__ == "__main__":
    unittest.main()
