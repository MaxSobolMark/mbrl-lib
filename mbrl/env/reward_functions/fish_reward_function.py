from .base_reward_function import BaseRewardFunction
from dm_control.utils import rewards
import torch
from .metaworld_reward_utils import tolerance


class FishRewardFunction(BaseRewardFunction):
    OBS_DIM = 24

    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        radii = self._env.unwrapped._env.physics.named.model.geom_size[
            ['mouth', 'target'], 0].sum()
        data = self.named.data
        mouth_to_target_global = data.geom_xpos['target']
        mouth_xpos = observation[:, -3:]
        mouth_to_target = 
        in_target = tolerance(torch.norm())
        return reward
