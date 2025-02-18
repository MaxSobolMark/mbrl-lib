from .base_reward_function import BaseRewardFunction
import torch
import numpy as np


class Reacher3DRewardFunction(BaseRewardFunction):
    OBS_DIM = 17

    def get_reward(self, observation: tf.Tensor, action: tf.Tensor):
        # goal = observation[:, 7:10]
        goal = self._env.unwrapped.goal
        obs_reward = -tf.reduce_sum(
            tf.square(self.get_ee_pos(observation) - goal), axis=1)
        action_cost = 0.01 * tf.reduce_sum(tf.square(action), axis=1)
        return obs_reward - action_cost

    def get_ee_pos(self, states, are_tensors=True):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = (
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4],
            states[:, 4:5], states[:, 5:6], states[:, 6:])
        if are_tensors:
            rot_axis = tf.concat([
                tf.cos(theta2) * tf.cos(theta1),
                tf.cos(theta2) * tf.sin(theta1), -tf.sin(theta2)
            ],
                                 axis=1)
            rot_perp_axis = tf.concat(
                [-tf.sin(theta1),
                 tf.cos(theta1),
                 tf.zeros(tf.shape(theta1))],
                axis=1)
            cur_end = tf.concat([
                0.1 * tf.cos(theta1) + 0.4 * tf.cos(theta1) * tf.cos(theta2),
                0.1 * tf.sin(theta1) + 0.4 * tf.sin(theta1) * tf.cos(theta2) -
                0.188, -0.4 * tf.sin(theta2)
            ],
                                axis=1)

            for length, hinge, roll in [(0.321, theta4, theta3),
                                        (0.16828, theta6, theta5)]:
                perp_all_axis = tf.linalg.cross(rot_axis, rot_perp_axis)
                x = tf.cos(hinge) * rot_axis
                y = tf.sin(hinge) * tf.sin(roll) * rot_perp_axis
                z = -tf.sin(hinge) * tf.cos(roll) * perp_all_axis
                new_rot_axis = x + y + z
                new_rot_perp_axis = tf.linalg.cross(new_rot_axis, rot_axis)
                new_rot_perp_axis = tf.where(
                    tf.less(tf.norm(new_rot_perp_axis, axis=1),
                            1e-30)[:, None], rot_perp_axis, new_rot_perp_axis)
                new_rot_perp_axis /= tf.norm(new_rot_perp_axis,
                                             axis=1,
                                             keepdims=True)
                rot_axis, rot_perp_axis, cur_end = (new_rot_axis,
                                                    new_rot_perp_axis,
                                                    cur_end +
                                                    length * new_rot_axis)
        else:
            rot_axis = np.concatenate([
                np.cos(theta2) * np.cos(theta1),
                np.cos(theta2) * np.sin(theta1), -np.sin(theta2)
            ],
                                      axis=1)
            rot_perp_axis = np.concatenate(
                [-np.sin(theta1),
                 np.cos(theta1),
                 np.zeros(theta1.shape)],
                axis=1)
            cur_end = np.concatenate([
                0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
                0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) -
                0.188, -0.4 * np.sin(theta2)
            ],
                                     axis=1)

            for length, hinge, roll in [(0.321, theta4, theta3),
                                        (0.16828, theta6, theta5)]:
                perp_all_axis = np.cross(rot_axis, rot_perp_axis)
                x = np.cos(hinge) * rot_axis
                y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
                z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
                new_rot_axis = x + y + z
                new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
                new_rot_perp_axis[np.linalg.norm(
                    new_rot_perp_axis, axis=1) < 1e-30] = rot_perp_axis[
                        np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
                new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis,
                                                    axis=1,
                                                    keepdims=True)
                rot_axis, rot_perp_axis, cur_end = (new_rot_axis,
                                                    new_rot_perp_axis,
                                                    cur_end +
                                                    length * new_rot_axis)

        return cur_end
