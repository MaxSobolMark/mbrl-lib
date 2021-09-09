import os
from typing import Tuple

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import torch
from gym import utils
from gym.envs.mujoco import mujoco_env


class Reacher3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task_id=None, hide_goal=False):
        self.viewer = None
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.goal = np.zeros(3)
        mujoco_env.MujocoEnv.__init__(
            self, os.path.join(dir_path, "assets/reacher3d.xml"), 2)
        self._task_id = task_id
        self._hide_goal = hide_goal
        if task_id is not None:
            self._rng = RandomState(MT19937(SeedSequence(task_id)))
            self.goal = self._rng.normal(loc=0, scale=0.1, size=[3])

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # print('[pets_reacher:22] ob[7:10]: ', ob[7:10])
        reward = -np.sum(
            np.square(Reacher3DEnv.get_EE_pos(ob[None]) - self.goal))
        reward -= 0.01 * np.square(a).sum()
        done = False
        return ob, reward, done, dict(reward_dist=0, reward_ctrl=0)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 270

    def reset_model(self):
        qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
        if self._task_id is not None:
            qpos[-3:] += self.goal
        else:
            qpos[-3:] += np.random.normal(loc=0, scale=0.1, size=[3])
            self.goal = qpos[-3:]
        qvel[-3:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        if not self._hide_goal:
            return np.concatenate([
                self.data.qpos.flat,
                self.data.qvel.flat[:-3],
            ])
        return np.concatenate([
            self.data.qpos.flat[:-3],
            self.data.qvel.flat[:-3],
        ])

    @staticmethod
    def get_EE_pos(states, are_tensors=False):
        theta1, theta2, theta3, theta4, theta5, theta6, _ = (
            states[:, :1],
            states[:, 1:2],
            states[:, 2:3],
            states[:, 3:4],
            states[:, 4:5],
            states[:, 5:6],
            states[:, 6:],
        )

        if not are_tensors:

            rot_axis = np.concatenate(
                [
                    np.cos(theta2) * np.cos(theta1),
                    np.cos(theta2) * np.sin(theta1),
                    -np.sin(theta2),
                ],
                axis=1,
            )
            rot_perp_axis = np.concatenate(
                [-np.sin(theta1),
                 np.cos(theta1),
                 np.zeros(theta1.shape)],
                axis=1)
            cur_end = np.concatenate(
                [
                    0.1 * np.cos(theta1) +
                    0.4 * np.cos(theta1) * np.cos(theta2),
                    0.1 * np.sin(theta1) +
                    0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
                    -0.4 * np.sin(theta2),
                ],
                axis=1,
            )

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
                rot_axis, rot_perp_axis, cur_end = (
                    new_rot_axis,
                    new_rot_perp_axis,
                    cur_end + length * new_rot_axis,
                )

            return cur_end
        else:
            rot_axis = torch.cat(
                [
                    torch.cos(theta2) * torch.cos(theta1),
                    torch.cos(theta2) * torch.sin(theta1),
                    -torch.sin(theta2),
                ],
                dim=1,
            )
            rot_perp_axis = torch.cat([
                -torch.sin(theta1),
                torch.cos(theta1),
                torch.zeros_like(theta1)
            ],
                                      dim=1)
            cur_end = torch.cat(
                [
                    0.1 * torch.cos(theta1) +
                    0.4 * torch.cos(theta1) * torch.cos(theta2),
                    0.1 * torch.sin(theta1) +
                    0.4 * torch.sin(theta1) * torch.cos(theta2) - 0.188,
                    -0.4 * torch.sin(theta2),
                ],
                dim=1,
            )

            for length, hinge, roll in [(0.321, theta4, theta3),
                                        (0.16828, theta6, theta5)]:
                perp_all_axis = torch.cross(rot_axis, rot_perp_axis)
                x = torch.cos(hinge) * rot_axis
                y = torch.sin(hinge) * torch.sin(roll) * rot_perp_axis
                z = -torch.sin(hinge) * torch.cos(roll) * perp_all_axis
                new_rot_axis = x + y + z
                new_rot_perp_axis = torch.cross(new_rot_axis, rot_axis)
                new_rot_perp_axis[torch.linalg.norm(
                    new_rot_perp_axis, dim=1) < 1e-30] = rot_perp_axis[
                        torch.linalg.norm(new_rot_perp_axis, dim=1) < 1e-30]
                new_rot_perp_axis /= torch.linalg.norm(new_rot_perp_axis,
                                                       dim=1,
                                                       keepdims=True)
                rot_axis, rot_perp_axis, cur_end = (
                    new_rot_axis,
                    new_rot_perp_axis,
                    cur_end + length * new_rot_axis,
                )

            return cur_end

    @staticmethod
    def get_reward(ob, action):
        # This is a bit tricky to implement, implement when needed
        print('NOT SUPPOSED TO RUN THIS!')
        raise NotImplementedError

    @staticmethod
    def forward_postprocess_fn(
        inputs: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor,
        min_logvar: torch.nn.parameter.Parameter
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean[..., 7:10] = inputs[..., 7:10]
        logvar[..., 7:10] = torch.full(logvar[..., 7:10].shape, -float('inf'))
        return mean, logvar
