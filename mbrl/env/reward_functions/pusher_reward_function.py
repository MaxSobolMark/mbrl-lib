from .base_reward_function import BaseRewardFunction
import torch


class PusherRewardFunction(BaseRewardFunction):
    OBS_DIM = 20

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        goal_pos = torch.tensor(self._env.ac_goal_pos).to(observation.device)

        to_w, og_w = 0.5, 1.25
        tip_pos, obj_pos = observation[:, 14:17], observation[:, 17:20]

        tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
        obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
        obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

        act_cost = 0.1 * (action**2).sum(axis=1)

        return -(obs_cost + act_cost).view(-1, 1).float()
