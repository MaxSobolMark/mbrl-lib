from .base_reward_function import BaseRewardFunction
import tensorflow as tf
import numpy as np
from .metaworld_reward_utils import tolerance


class FaucetOpenRewardFunction(BaseRewardFunction):
    OBS_DIM = 39

    def get_reward(self, observations: tf.Tensor, actions: tf.Tensor):
        gripper = observations[:, :3]
        obj = observations[:, 4:7] + np.array([-.04, .0, .03])
        tcp = gripper
        # tcp = self.tcp_center
        target = self._env.unwrapped._target_pos.copy()

        target_to_obj = (obj - target)
        target_to_obj = tf.norm(target_to_obj, axis=-1)
        target_to_obj_init = (self._env.unwrapped.obj_init_pos - target)
        target_to_obj_init = tf.norm(target_to_obj_init, axis=-1)

        in_place = tolerance(
            target_to_obj,
            bounds=(0, self._env.unwrapped._target_radius),
            margin=abs(target_to_obj_init -
                       self._env.unwrapped._target_radius),
            sigmoid='long_tail',
        )

        faucet_reach_radius = 0.01
        tcp_to_obj = tf.norm(obj - tcp, axis=-1)
        tcp_to_obj_init = tf.norm(self._env.unwrapped.obj_init_pos -
                                  self._env.unwrapped.init_tcp)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0, faucet_reach_radius),
            margin=abs(tcp_to_obj_init - faucet_reach_radius),
            sigmoid='gaussian',
        )

        reward = 2 * reach + 3 * in_place

        reward *= 2
        reward = tf.where(target_to_obj <= self._env.unwrapped._target_radius,
                          10., reward)
        # reward = 10 if target_to_obj <= self._env.unwrapped._target_radius else reward

        return reward
