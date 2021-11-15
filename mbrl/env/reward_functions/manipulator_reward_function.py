from .base_reward_function import BaseRewardFunction
from dm_control.utils import rewards
import torch
from .metaworld_reward_utils import tolerance

_CLOSE = .01  # (Meters) Distance below which a thing is considered close.


class ManipulatorRewardFunction(BaseRewardFunction):
    OBS_DIM = 44

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        batch_size = observation.shape[0]
        ball_xz_position = observation[:, -11:-9]
        ball_position = torch.cat([
            ball_xz_position[:, 0][:, None],
            torch.zeros([batch_size, 1]).to(device),
            ball_xz_position[:, 1][:, None]
        ],
                                  dim=-1)
        target_ball_position = (self._env.unwrapped._env.physics.named.data.
                                site_xpos['target_ball'])
        site_diff = ball_position - torch.Tensor(target_ball_position).to(
            device)
        site_distance = torch.norm(site_diff, dim=-1)
        reward = tolerance(site_distance, (0, _CLOSE),
                           torch.Tensor([_CLOSE * 2]).to(device),
                           device=device)
        return reward
