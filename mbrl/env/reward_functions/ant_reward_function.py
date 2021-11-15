from .base_reward_function import BaseRewardFunction
import torch


class AntRewardFunction(BaseRewardFunction):
    OBS_DIM = 29  # Base 17 + velocity.

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        goal = torch.Tensor([self._env._goal]).to(device)
        direct = torch.Tensor([torch.cos(goal), torch.sin(goal)]).to(device)
        xy_velocity = observation[:, -2:]
        x_velocity, y_velocity = xy_velocity.split(1, dim=-1)

        # xposbefore = self.get_body_com("torso")[0]
        # self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]

        # forward_reward = (xposafter - xposbefore)/self.dt
        angle_reward = torch.matmul(xy_velocity, direct)
        ctrl_cost = .5 * torch.square(action).sum(dim=-1)
        survive_reward = 1.0
        reward = angle_reward - ctrl_cost + survive_reward
        return reward
