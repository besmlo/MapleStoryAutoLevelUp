class State:
    """Base behavior shared by the bot's per-frame operating modes."""

    def __init__(self, name, bot):
        self.name = name
        self.bot = bot

    def on_enter(self):
        """Hook called when this state becomes active."""

    def on_exit(self):
        """Hook called before leaving this state."""

    def on_frame(self):
        raise NotImplementedError

    def recover_if_stuck(self):
        if self.bot.is_player_stuck():
            self.bot.update_cmd_by_random()

    def send_current_command(self):
        command = " ".join(
            (
                self.bot.cmd_move_x,
                self.bot.cmd_move_y,
                self.bot.cmd_action,
            )
        )
        self.bot.kb.set_command(command)
