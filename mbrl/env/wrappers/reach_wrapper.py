""""""
import numpy as np
from gym import ObservationWrapper, spaces


class ReachWrapper(ObservationWrapper):
    def _get_obs_indices(self, obs_dim):
        indices = np.arange(obs_dim)
        goal = indices[-3:]
        indices = indices[:-3]
        indices1, indices2 = np.split(indices, 2)
        return np.concatenate([indices1[:4], indices2[:4], goal])

    def __init__(self, env):
        super(ReachWrapper, self).__init__(env)
        indices = self._get_obs_indices(
            np.prod(env.observation_space.low.shape))
        low = env.observation_space.low[indices]
        high = env.observation_space.high[indices]
        self.observation_space = spaces.Box(low, high)

    def observation(self, obs):
        # take out the object position.
        indices = self._get_obs_indices(np.prod(obs.shape))
        obs = obs[indices]
        return obs
