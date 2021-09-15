from .base_reward_function import BaseRewardFunction
import torch
import numpy as np
from .metaworld_reward_utils import tolerance, hamacher_product


class DrawerCloseRewardFunction(BaseRewardFunction):
    OBS_DIM = 39

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        gripper = observation[:, :3]
        obj = observation[:, 4:7]

        tcp = gripper  # self.tcp_center
        target = torch.Tensor(
            self._env.unwrapped._target_pos.copy()).to(device)

        target_to_obj = (obj - target)
        target_to_obj = torch.norm(target_to_obj, dim=-1)
        target_to_obj_init = (
            torch.Tensor(self._env.unwrapped.obj_init_pos).to(device) - target)
        target_to_obj_init = torch.norm(target_to_obj_init, dim=-1)

        in_place = tolerance(
            target_to_obj,
            bounds=(0, self._env.unwrapped.TARGET_RADIUS),
            margin=torch.Tensor([
                abs(target_to_obj_init - self._env.unwrapped.TARGET_RADIUS)
            ]).to(device),
            sigmoid='long_tail',
            device=device,
        )

        handle_reach_radius = 0.005
        tcp_to_obj = torch.norm(obj - tcp, dim=-1)
        tcp_to_obj_init = np.linalg.norm(self._env.unwrapped.obj_init_pos -
                                         self._env.unwrapped.init_tcp,
                                         axis=-1)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0, handle_reach_radius),
            margin=torch.Tensor([abs(tcp_to_obj_init - handle_reach_radius)
                                 ]).to(device),
            sigmoid='gaussian',
            device=device,
        )
        gripper_closed = torch.clamp(action[:, -1], 0, 1)

        reach = hamacher_product(reach, gripper_closed, device)

        reward = hamacher_product(reach, in_place, device)
        reward = torch.where(
            target_to_obj <= self._env.unwrapped.TARGET_RADIUS + 0.015,
            torch.ones(1).to(device), reward * 10)
        # if target_to_obj <= self._env.unwrapped.TARGET_RADIUS + 0.015:
        #     reward = 1.

        # reward *= 10

        return reward
