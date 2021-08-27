from .base_reward_function import BaseRewardFunction
import torch


class HalfCheetahJumpRewardFunction(BaseRewardFunction):
    OBS_DIM = 18

    def __init__(self, env):
        super(HalfCheetahJumpRewardFunction, self).__init__(env)
        self._z_init = self._env.unwrapped.sim.data.qpos[1]

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor):
        reward_ctrl = -0.1 * torch.sum(torch.square(action), dim=-1)
        reward_run = observation[:, 0]
        z = observation[:, 1]
        reward_jump = 15 * torch.maximum(z - self._z_init,
                                         torch.FloatTensor([0]))
        return reward_run + reward_ctrl + reward_jump
