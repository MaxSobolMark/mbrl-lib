from .base_reward_function import BaseRewardFunction
import torch
import numpy as np
from .metaworld_reward_utils import tolerance


class FaucetCloseRewardFunction(BaseRewardFunction):
    OBS_DIM = 39

    def get_reward(self, observations: torch.Tensor, actions: torch.Tensor,
                   device: str):
        gripper = observations[:, :3]
        obj = observations[:, 4:7]
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
            bounds=(0, self._env.unwrapped._target_radius),
            margin=torch.Tensor([
                abs(target_to_obj_init - self._env.unwrapped._target_radius)
            ]).to(device),
            sigmoid='long_tail',
            device=device,
        )

        faucet_reach_radius = 0.01
        tcp_to_obj = torch.norm(obj - tcp, dim=-1)
        tcp_to_obj_init = np.linalg.norm(self._env.unwrapped.obj_init_pos -
                                         self._env.unwrapped.init_tcp,
                                         axis=-1)
        reach = tolerance(tcp_to_obj,
                          bounds=(0, faucet_reach_radius),
                          margin=torch.Tensor([
                              abs(tcp_to_obj_init - faucet_reach_radius)
                          ]).to(device),
                          sigmoid='gaussian',
                          device=device)

        reward = 2 * reach + 3 * in_place
        reward *= 2
        reward = torch.where(
            target_to_obj <= self._env.unwrapped._target_radius,
            torch.Tensor([10.]).to(device), reward)
        # reward = 10 if target_to_obj <= self._target_radius else reward

        return reward
