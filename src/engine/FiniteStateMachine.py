# Local import
from src.states.base_state import State
from src.utils.logger import logger


class FiniteStateMachine:
    '''
    Finite State Machine
    '''
    def __init__(self):
        self.states = {}
        self.state = None

    def add_state(self, state: State):
        self.states[state.name] = state

    def set_state(self, state_name):
        next_state = self.states.get(state_name)
        if next_state is None:
            logger.error(f"[FSM] Unexpected state: {state_name}")
            return False
        if self.state is not None:
            self.state.on_exit()
        self.state = next_state
        self.state.on_enter()
        return True

    def run_frame(self):
        if self.state is None:
            raise RuntimeError("Bot state has not been initialized")
        self.state.on_frame()
