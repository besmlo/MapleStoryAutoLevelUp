from src.states.base_state import State


class HuntingState(State):
    def on_frame(self):
        # Get command from route map
        self.bot.update_cmd_by_route()

        # Check if reach goal on route map
        self.bot.check_reach_goal()

        # Get attack command by detecting mobs near players
        self.bot.update_cmd_by_mob_detection()

        self.recover_if_stuck()
        self.send_current_command()
