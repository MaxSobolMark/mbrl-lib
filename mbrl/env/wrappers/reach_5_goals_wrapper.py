""""""
import numpy as np
from gym import Wrapper, spaces
import metaworld
from numpy.random import MT19937, RandomState, SeedSequence


class Reach5GoalsWrapper(Wrapper):
    def _get_obs_indices(self, obs_dim):
        indices = np.arange(obs_dim)
        goal = indices[-3:]
        indices = indices[:-3]
        indices1, indices2 = np.split(indices, 2)
        return np.concatenate([indices1[:4], indices2[:4], goal])

    def __init__(self, env):
        super(Reach5GoalsWrapper, self).__init__(env)
        indices = self._get_obs_indices(
            np.prod(env.observation_space.low.shape))
        low = env.observation_space.low[indices]
        high = env.observation_space.high[indices]
        self.observation_space = spaces.Box(low, high)
        mt1 = metaworld.MT1('reach-v2')
        self._tasks = mt1.train_tasks[:5]
        self._random = RandomState(MT19937(SeedSequence(123)))

    def observation(self, obs):
        indices = self._get_obs_indices(np.prod(obs.shape))
        obs = obs[indices]
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.observation(observation)
        return observation, reward, done, info

    def reset(self):
        task_index = self._random.randint(0, 5)
        self.env.set_task(self._tasks[task_index])
        return self.observation(self.env.reset())
