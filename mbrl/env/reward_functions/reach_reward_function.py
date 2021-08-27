from .base_reward_function import BaseRewardFunction
import tensorflow as tf
import numpy as np
from .metaworld_reward_utils import tolerance


class ReachRewardFunction(BaseRewardFunction):
    OBS_DIM = 39

    def get_reward(self, observation: tf.Tensor, action: tf.Tensor):
        _TARGET_RADIUS = np.float32(0.05)
        # tcp = self._env.unwrapped.tcp_center
        gripper = observation[:, :3]
        tcp = gripper
        obj = observation[:, 4:7]
        target = self._env.unwrapped._target_pos

        tcp_to_target = tf.norm(tcp - target, axis=-1)

        in_place_margin = tf.norm(self._env.unwrapped.hand_init_pos - target)
        in_place = tolerance(
            tcp_to_target,
            bounds=(np.float32(0.), _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid='long_tail',
        )

        return 10 * in_place
