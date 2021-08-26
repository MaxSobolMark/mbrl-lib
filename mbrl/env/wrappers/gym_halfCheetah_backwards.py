"""Reward wrapper that gives rewards for positive change in z axis.
   Based on MOPO: https://arxiv.org/abs/2005.13239"""

import numpy as np
from gym import Wrapper


class HalfCheetahBackwardsWrapper(Wrapper):
    def step(self, action):
        observation, original_reward, done, info = self.env.step(action)
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = -observation[0]
        reward = reward_run + reward_ctrl
        return observation, reward, done, info
        self.env.prev_qpos = np.copy(self.env.sim.data.qpos.flat)
        self.env.do_simulation(action, self.env.frame_skip)
        ob = self.env._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = -ob[0]
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}
