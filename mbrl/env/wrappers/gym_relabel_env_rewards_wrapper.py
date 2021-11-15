"""Reward wrapper that gives rewards for positive change in z axis.
   Based on MOPO: https://arxiv.org/abs/2005.13239"""

from typing import Callable
from gym import Wrapper
import torch


class RelabelEnvRewardsWrapper(Wrapper):
    def __init__(self, env, reward_function: Callable, device: str):
        super(RelabelEnvRewardsWrapper, self).__init__(env)
        self._reward_function = reward_function
        self._device = device

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = self._reward_function(
            torch.Tensor([action]).to(self._device),
            torch.Tensor([observation]).to(self._device),
            self._device).cpu().detach().numpy()
        return observation, float(reward), done, info
