"""Reward wrapper that gives rewards for positive change in z axis.
   Based on MOPO: https://arxiv.org/abs/2005.13239"""

from gym import Wrapper


class JumpWrapper(Wrapper):
    def __init__(self, env):
        super(JumpWrapper, self).__init__(env)
        self._z_init = self.env.sim.data.qpos[1]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        z = self.env.sim.data.qpos[1]
        reward = reward + 15 * max(z - self._z_init, 0)
        return observation, reward, done, info
