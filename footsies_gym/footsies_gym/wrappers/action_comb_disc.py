import gymnasium as gym
from gymnasium import spaces


class FootsiesActionCombinationsDiscretized(gym.ActionWrapper):
    """
    Discretizes the FOOTSIES actions, which are a tuple of four boolean values, into a single integer representing all possible combinations of those boolean values

    Actions: (left, right, attack, combo_toggle)
    For an action represented by an integer, the respective tuple is equal to its first (rightmost) 4 bits, read from right to left.
    This is compatible with the game's internal representation of the players' input (first 3 bits), with the 4th bit for combo toggle.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(2**4)  # 16 possible actions

    def action(self, act):
        return ((act & 1) != 0, (act & 2) != 0, (act & 4) != 0, (act & 8) != 0)
