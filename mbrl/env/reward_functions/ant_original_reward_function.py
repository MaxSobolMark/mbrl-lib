from .base_reward_function import BaseRewardFunction
import torch


class AntOriginalRewardFunction(BaseRewardFunction):
    OBS_DIM = 27  # Base 17 + velocity.

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        velocity_cost = -observation[:, 13]
        height_cost = 3 * torch.square(observation[:, 0] - 0.57)  # the height
        action_cost = 0.1 * torch.square(action).sum(dim=1)
        return velocity_cost + height_cost - action_cost
