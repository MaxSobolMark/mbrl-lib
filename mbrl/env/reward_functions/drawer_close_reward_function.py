from .base_reward_function import BaseRewardFunction
import tensorflow as tf
import numpy as np
from .metaworld_reward_utils import tolerance, hamacher_product


class DrawerCloseRewardFunction(BaseRewardFunction):
    OBS_DIM = 39

    def get_reward(self, observation: tf.Tensor, action: tf.Tensor):
        gripper = observation[:, :3]
        obj = observation[:, 4:7]

        tcp = gripper  # self.tcp_center
        target = self._env.unwrapped._target_pos.copy()

        target_to_obj = (obj - target)
        target_to_obj = tf.norm(target_to_obj, axis=-1)
        target_to_obj_init = (self._env.unwrapped.obj_init_pos - target)
        target_to_obj_init = tf.norm(target_to_obj_init, axis=-1)

        in_place = tolerance(
            target_to_obj,
            bounds=(0, self._env.unwrapped.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self._env.unwrapped.TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_reach_radius = 0.005
        tcp_to_obj = tf.norm(obj - tcp, axis=-1)
        tcp_to_obj_init = tf.norm(self._env.unwrapped.obj_init_pos -
                                  self._env.unwrapped.init_tcp,
                                  axis=-1)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0, handle_reach_radius),
            margin=abs(tcp_to_obj_init - handle_reach_radius),
            sigmoid='gaussian',
        )
        gripper_closed = tf.math.minimum(
            tf.math.maximum(tf.cast(0, tf.float32), action[:, -1]),
            tf.cast(1, tf.float32))

        reach = hamacher_product(reach, gripper_closed)

        reward = hamacher_product(reach, in_place)
        reward = tf.where(
            target_to_obj <= self._env.unwrapped.TARGET_RADIUS + 0.015, 1.,
            reward * 10)
        # if target_to_obj <= self._env.unwrapped.TARGET_RADIUS + 0.015:
        #     reward = 1.

        # reward *= 10

        return reward
