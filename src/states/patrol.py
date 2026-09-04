import time

# Local import
from src.states.base_state import State


class PatrolState(State):
    def __init__(self, name, bot):
        super().__init__(name, bot)
        self.is_patrol_to_left = True # Patrol direction flag
        self.patrol_turn_point_cnt = 0 # Patrol tuning back counter

    def on_frame(self):
        x, _ = self.bot.loc_player
        _, w = self.bot.img_frame.shape[:2]
        loc_player_ratio = float(x)/float(w)
        left_ratio, right_ratio = self.bot.cfg["patrol"]["range"]

        # Check if we need to change patrol direction
        if self.is_patrol_to_left and loc_player_ratio < left_ratio:
            self.patrol_turn_point_cnt += 1
        elif (not self.is_patrol_to_left) and loc_player_ratio > right_ratio:
            self.patrol_turn_point_cnt += 1

        if self.patrol_turn_point_cnt > self.bot.cfg["patrol"]["turn_point_thres"]:
            self.is_patrol_to_left = not self.is_patrol_to_left
            self.patrol_turn_point_cnt = 0

        # Update cmd_move_x
        if self.is_patrol_to_left:
            self.bot.cmd_move_x = "left"
        else:
            self.bot.cmd_move_x = "right"

        # Update attack command by detecting mobs near players
        self.bot.update_cmd_by_mob_detection()

        # Update attack command by periodically attacking
        if time.time() - self.bot.t_last_attack > \
            self.bot.cfg["patrol"]["patrol_attack_interval"]:
            self.bot.cmd_action = "attack"
            self.bot.t_last_attack = time.time()

        self.recover_if_stuck()
        self.send_current_command()
