from .base_reward_function import BaseRewardFunction
import torch


class HalfCheetahBackwardsRewardFunction(BaseRewardFunction):
    OBS_DIM = 18

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        reward_ctrl = -0.1 * torch.sum(torch.square(action), dim=-1)
        reward_run = -observation[:, 0]
        return reward_run + reward_ctrl
