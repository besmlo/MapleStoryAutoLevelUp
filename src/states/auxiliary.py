# Local import
from src.states.base_state import State


class AuxiliaryState(State):
    def on_frame(self):
        """Auxiliary mode observes the game without issuing movement."""
