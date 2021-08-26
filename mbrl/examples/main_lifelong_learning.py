# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Tuple, Optional
import gym
import hydra
import numpy as np
import omegaconf
from omegaconf import OmegaConf
import torch
import wandb

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.util.mujoco as mujoco_util
import mbrl.types
from mbrl.env.wrappers.multitask_wrapper import MultitaskWrapper


def make_env(
    env_name: str, wrappers: List, learned_rewards: bool,
    cfg: omegaconf.DictConfig
) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    if "gym___" in env_name:
        import mbrl.env

        env = gym.make(env_name.split("___")[1])
        term_fn = getattr(mbrl.env.termination_fns,
                          cfg.lifelong_learning.overrides.term_fn)
        if hasattr(cfg.lifelong_learning.overrides, "reward_fn"
                   ) and cfg.lifelong_learning.overrides.reward_fn is not None:
            reward_fn = getattr(mbrl.env.reward_fns,
                                cfg.lifelong_learning.overrides.reward_fn)
        else:
            reward_fn = getattr(mbrl.env.reward_fns,
                                cfg.lifelong_learning.overrides.term_fn, None)
    else:
        import mbrl.env.mujoco_envs

        if env_name == "cartpole_continuous":
            env = mbrl.env.cartpole_continuous.CartPoleEnv()
            term_fn = mbrl.env.termination_fns.cartpole
            reward_fn = mbrl.env.reward_fns.cartpole
        elif env_name == "pets_halfcheetah":
            env = mbrl.env.mujoco_envs.HalfCheetahEnv()
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = getattr(mbrl.env.reward_fns, "halfcheetah", None)
        elif env_name == "pets_reacher":
            env = mbrl.env.mujoco_envs.Reacher3DEnv()
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = getattr(mbrl.env.reward_fns, 'reacher', None)
        elif env_name == "pets_pusher":
            env = mbrl.env.mujoco_envs.PusherEnv()
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = mbrl.env.reward_fns.pusher
        elif env_name == "ant_truncated_obs":
            env = mbrl.env.mujoco_envs.AntTruncatedObsEnv()
            term_fn = mbrl.env.termination_fns.ant
            reward_fn = None
        elif env_name == "humanoid_truncated_obs":
            env = mbrl.env.mujoco_envs.HumanoidTruncatedObsEnv()
            term_fn = mbrl.env.termination_fns.ant
            reward_fn = None
        else:
            raise ValueError("Invalid environment string.")
        env = gym.wrappers.TimeLimit(
            env,
            max_episode_steps=cfg.overrides.lifelong_learning.get(
                "trial_length", 1000))
    for wrapper_name in wrappers:
        env = hydra.utils.instantiate(wrapper_name, env=env)

    learned_rewards = cfg.overrides.lifelong_learning.get(
        "learned_rewards", True)
    if learned_rewards:
        reward_fn = None

    if cfg.seed is not None:
        env.seed(cfg.seed)
        env.observation_space.seed(cfg.seed + 1)
        env.action_space.seed(cfg.seed + 2)

    return env, term_fn, reward_fn


@hydra.main(config_path="conf", config_name="main_lifelong_learning")
def run(cfg: omegaconf.DictConfig):
    wandb.init(project='fsrl-mbrl', config=OmegaConf.to_container(cfg))
    lifelong_learning_envs = []
    lifelong_learning_task_names = []
    lifelong_learning_termination_fns = []
    lifelong_learning_reward_fns = []
    num_tasks = len(cfg.overrides.lifelong_learning.envs)
    for i, env_cfg in enumerate(cfg.overrides.lifelong_learning.envs):
        print('[main_lifelong_learning:94] wrappers: ', env_cfg.wrappers)
        env, term_fn, reward_fn = make_env(
            env_cfg.env_name, env_cfg.wrappers,
            cfg.overrides.lifelong_learning.learned_rewards, cfg)
        env = MultitaskWrapper(env, i, num_tasks)
        lifelong_learning_envs.append(env)
        lifelong_learning_task_names.append(env_cfg.task_name)
        lifelong_learning_termination_fns.append(term_fn)
        lifelong_learning_reward_fns.append(reward_fn)

    print(
        f'[main_lifelong_learning:97] lifelong_learning_envs: {lifelong_learning_envs}'
    )
    print(
        f'[main_lifelong_learning:98] lifelong_learning_termination_fns: {lifelong_learning_termination_fns}'
    )
    print(
        f'[main_lifelong_learning:102] lifelong_learning_reward_fns: {lifelong_learning_reward_fns}'
    )
    # np.random.seed(cfg.seed)
    # torch.manual_seed(cfg.seed)
    # if cfg.algorithm.name == "pets":
    #     return pets.train(env, term_fn, reward_fn, cfg)
    # if cfg.algorithm.name == "mbpo":
    #     test_env, *_ = mujoco_util.make_env(cfg)
    #     return mbpo.train(env, test_env, term_fn, cfg)


if __name__ == "__main__":
    run()
