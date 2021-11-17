# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)


def inverted_pendulum(act: torch.Tensor,
                      next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(
        -1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act**2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)


def reacher(act: torch.Tensor,
            next_obs: torch.Tensor,
            device: str = '') -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    # print('[reward_fns:49] next_obs.shape: ', next_obs.shape)
    goals = next_obs[:, 7:10]
    # print('[reward_fns:50] goals: ', goals)
    from mbrl.env.pets_reacher import Reacher3DEnv
    reward = -torch.sum(torch.square(
        Reacher3DEnv.get_EE_pos(next_obs, are_tensors=True) - goals),
                        dim=1)
    reward -= 0.01 * torch.square(act).sum(dim=1)
    return reward.view(-1, 1)


def walker2d(act: torch.Tensor,
             next_obs: torch.Tensor,
             device: str = '') -> torch.Tensor:
    velocity_cost = -next_obs[:, 8]  # the qvel for the root-x joint
    height_cost = 3 * torch.square(next_obs[:, 0] - 1.3)  # the height
    action_cost = 0.1 * torch.square(act).sum(dim=-1)
    return (velocity_cost + height_cost - action_cost).view(-1, 1)
