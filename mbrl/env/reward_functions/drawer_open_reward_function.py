from .base_reward_function import BaseRewardFunction
import tensorflow as tf
import numpy as np
from .metaworld_reward_utils import tolerance


class DrawerOpenRewardFunction(BaseRewardFunction):
    OBS_DIM = 39

    def get_reward(self, observation: tf.Tensor, action: tf.Tensor):
        gripper = observation[:, :3]
        handle = observation[:, 4:7]

        handle_error = tf.norm(handle - self._env.unwrapped._target_pos,
                               axis=-1)

        reward_for_opening = tolerance(handle_error,
                                       bounds=(0, 0.02),
                                       margin=self._env.unwrapped.maxDist,
                                       sigmoid='long_tail')
        handle_pos_init = self._env.unwrapped._target_pos + np.array(
            [.0, self._env.unwrapped.maxDist, .0])
        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward
        scale = np.array([3., 3., 1.])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init -
                              self._env.unwrapped.init_tcp) * scale

        reward_for_caging = tolerance(tf.norm(gripper_error),
                                      bounds=(0, 0.01),
                                      margin=tf.norm(gripper_error_init),
                                      sigmoid='long_tail')

        reward = reward_for_caging + reward_for_opening
        reward *= 5.0

        return reward
