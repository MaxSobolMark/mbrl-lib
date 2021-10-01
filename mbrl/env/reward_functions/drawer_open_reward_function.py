from .base_reward_function import BaseRewardFunction
import torch
import numpy as np
from .metaworld_reward_utils import tolerance


class DrawerOpenRewardFunction(BaseRewardFunction):
    OBS_DIM = 39

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        gripper = observation[:, :3]
        handle = observation[:, 4:7]

        handle_error = torch.norm(
            handle - torch.Tensor(self._env.unwrapped._target_pos).to(device),
            dim=-1)

        reward_for_opening = tolerance(
            handle_error,
            bounds=(0, 0.02),
            margin=torch.Tensor([self._env.unwrapped.maxDist]).to(device),
            sigmoid='long_tail',
            device=device)
        handle_pos_init = torch.Tensor(
            self._env.unwrapped._target_pos).to(device) + torch.Tensor(
                [.0, self._env.unwrapped.maxDist, .0]).to(device)
        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward
        scale = torch.Tensor([3., 3., 1.]).to(device)
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - torch.Tensor(
            self._env.unwrapped.init_tcp).to(device)) * scale

        reward_for_caging = tolerance(
            torch.norm(gripper_error, dim=-1),
            bounds=(0, 0.01),
            margin=torch.norm(gripper_error_init).to(device),
            sigmoid='long_tail',
            device=device)

        reward = reward_for_caging + reward_for_opening
        reward *= 5.0

        return reward
