from .base_reward_function import BaseRewardFunction
import torch


class SwimmerOriginalRewardFunction(BaseRewardFunction):
    OBS_DIM = 8  # Base 17 + velocity.

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        velocity_cost = -observation[:, 3]
        action_cost = 0.0001 * torch.square(action).sum(dim=1)
        return velocity_cost - action_cost
