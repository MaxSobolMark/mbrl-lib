from .base_reward_function import BaseRewardFunction
import torch


class SwimmerRewardFunction(BaseRewardFunction):
    OBS_DIM = 8  # Base 17 + velocity.

    def get_reward(self,
                   observation: torch.Tensor,
                   action: torch.Tensor,
                   device: str,
                   goal: float = None):
        assert goal is not None
        goal = torch.Tensor([goal]).to(device)
        direct = torch.Tensor([torch.cos(goal), torch.sin(goal)]).to(device)
        xy_velocity = observation[..., 3:5]
        x_velocity, y_velocity = xy_velocity.split(1, dim=-1)

        angle_reward = torch.matmul(xy_velocity, direct)
        ctrl_cost = .0001 * torch.square(action).sum(dim=-1)
        reward = angle_reward - ctrl_cost
        return reward
