"""Metaworld environment wrapper that hides the goal position."""
from gym import ObservationWrapper, spaces


class MetaworldHideGoalWrapper(ObservationWrapper):
    def __init__(self, env):
        super(MetaworldHideGoalWrapper, self).__init__(env)
        low = env.observation_space.low[:-3]
        high = env.observation_space.high[:-3]
        self.observation_space = spaces.Box(low, high)

    def observation(self, observation):
        return observation[:-3]
