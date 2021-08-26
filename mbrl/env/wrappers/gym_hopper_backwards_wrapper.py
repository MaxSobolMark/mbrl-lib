"""Reward wrapper that gives rewards for positive change in z axis.
   Based on MOPO: https://arxiv.org/abs/2005.13239"""

import numpy as np
from gym import Wrapper


class HopperBackwardsWrapper(Wrapper):
    def step(self, action):
        pos_before = self.env.sim.data.qpos[0]
        observation, reward, done, info = self.env.step(action)
        pos_after = self.env.sim.data.qpos[0]
        alive_bonus = 1.
        reward = (pos_before - pos_after) / self.env.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()

        return observation, reward, done, info
