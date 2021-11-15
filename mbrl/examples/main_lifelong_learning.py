# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Tuple, Optional
from functools import partial
import re
import gym
import hydra
import numpy as np
import omegaconf
from omegaconf import OmegaConf
import torch
import wandb

# import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.lifelong_learning_pets as lifelong_learning_pets
import mbrl.algorithms.fsrl as fsrl
import mbrl.util.mujoco as mujoco_util
import mbrl.types
from mbrl.env.wrappers.multitask_wrapper import MultitaskWrapper
from mbrl.util.lifelong_learning import make_task_name_to_index_map
from mbrl.env.reward_functions import DOMAIN_TO_REWARD_FUNCTION


def preprocess_dmc_env(env, env_cfg: omegaconf.DictConfig):
    unwrapped_env = env.unwrapped._env
    original_initialize_episode = unwrapped_env.task.initialize_episode
    hide_goal = env_cfg.get('hide_goal', False)
    add_mouth_xpos = env_cfg.get('add_mouth_xpos', True)
    target = env_cfg.get('target', None)
    target_object_name = env_cfg.get('target_object_name', 'target')

    def initialize_episode_wrapper(physics):
        original_initialize_episode(physics)
        if target is not None:
            physics.named.model.geom_pos[target_object_name] = target
            physics.named.model.body_pos[target_object_name] = target

    unwrapped_env.task.initialize_episode = initialize_episode_wrapper
    original_get_observation = unwrapped_env.task.get_observation

    def get_observation_wrapper(physics):
        obs = original_get_observation(physics)
        if hide_goal:
            del obs['target']
        if add_mouth_xpos:
            obs['mouth_xpos'] = physics.named.data.geom_xpos['mouth']
        return obs

    unwrapped_env.task.get_observation = get_observation_wrapper
    from mbrl.third_party.dmc2gym.wrappers import _spec_to_box
    env.observation_space = _spec_to_box(env.observation_spec().values())
    from dm_control.suite import manipulator
    manipulator._P_IN_HAND = .5
    return env


def make_env(
    env_name: str,
    wrappers: List,
    learned_rewards: bool,
    cfg: omegaconf.DictConfig,
    env_cfg: omegaconf.DictConfig,
) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    if "dmcontrol___" in cfg.overrides.env:
        import mbrl.third_party.dmc2gym as dmc2gym
        import mbrl.env

        domain, task = env_name.split("___")[1].split("--")
        #term_fn, reward_fn = _get_term_and_reward_fn(cfg)
        term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
        env = dmc2gym.make(domain_name=domain, task_name=task)
        if domain.lower() in DOMAIN_TO_REWARD_FUNCTION:
            reward_fn = DOMAIN_TO_REWARD_FUNCTION[domain.lower()](env)
        elif hasattr(cfg.overrides,
                     "reward_fn") and cfg.overrides.reward_fn is not None:
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.reward_fn)
    elif "gym___" in env_name:
        import mbrl.env
        gym_env_name = env_name.split("___")[1]
        env = gym.make(gym_env_name)
        term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
        if env_cfg.reward_fn.lower() in DOMAIN_TO_REWARD_FUNCTION:
            reward_fn = DOMAIN_TO_REWARD_FUNCTION[env_cfg.reward_fn.lower()](
                env)
        elif hasattr(cfg.overrides,
                     "reward_fn") and cfg.overrides.reward_fn is not None:
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.reward_fn)
        else:
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn,
                                None)
    else:
        import mbrl.env.mujoco_envs

        if env_name == "cartpole_continuous":
            env = mbrl.env.cartpole_continuous.CartPoleEnv()
            term_fn = mbrl.env.termination_fns.cartpole
            reward_fn = mbrl.env.reward_fns.cartpole
        elif env_name == "pets_halfcheetah":
            env = mbrl.env.mujoco_envs.HalfCheetahEnv()
            term_fn = mbrl.env.termination_fns.no_termination
            # reward_fn = getattr(mbrl.env.reward_fns, "halfcheetah", None)
            print('------------------------------------')
            print('[main_lifelong_learning:54] env_cfg: ', env_cfg)
            reward_fn = DOMAIN_TO_REWARD_FUNCTION[env_cfg.reward_fn](env)
            print('[main_lifelong_learning:56] reward_fn: ', reward_fn)
            print('------------------------------------')
        elif env_name == "pets_reacher":
            env_kwargs = env_cfg.get('env_kwargs', {})
            env = mbrl.env.mujoco_envs.Reacher3DEnv(**env_kwargs)
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = getattr(mbrl.env.reward_fns, 'reacher', None)
        elif env_name == "pets_pusher":
            env_kwargs = env_cfg.get('env_kwargs', {})
            env = mbrl.env.mujoco_envs.PusherEnv(**env_kwargs)
            term_fn = mbrl.env.termination_fns.no_termination
            # reward_fn = mbrl.env.reward_fns.pusher
            reward_fn = DOMAIN_TO_REWARD_FUNCTION[env_cfg.reward_fn](env)
        elif env_name == "ant_truncated_obs":
            env_kwargs = env_cfg.get('env_kwargs', {})
            env = mbrl.env.mujoco_envs.AntTruncatedObsEnv(**env_kwargs)
            #term_fn = mbrl.env.termination_fns.ant
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = DOMAIN_TO_REWARD_FUNCTION[env_cfg.reward_fn](env)
        elif env_name == "ant_truncated_obs_original":
            env_kwargs = env_cfg.get('env_kwargs', {})
            env = mbrl.env.mujoco_envs.AntTruncatedObsOriginalEnv(**env_kwargs)
            #term_fn = mbrl.env.termination_fns.ant
            term_fn = mbrl.env.termination_fns.no_termination
            reward_fn = DOMAIN_TO_REWARD_FUNCTION[env_cfg.reward_fn](env)
        elif env_name == "humanoid_truncated_obs":
            env = mbrl.env.mujoco_envs.HumanoidTruncatedObsEnv()
            term_fn = mbrl.env.termination_fns.ant
            reward_fn = None
        else:
            raise ValueError("Invalid environment string.")
        env = gym.wrappers.TimeLimit(env,
                                     max_episode_steps=cfg.overrides.get(
                                         "trial_length", 1000))
    for wrapper_name in wrappers:
        env = hydra.utils.instantiate(wrapper_name, env=env)

    learned_rewards = cfg.overrides.get("learned_rewards", True)
    if learned_rewards:
        reward_fn = None
    else:
        reward_fn = partial(reward_fn, **env_cfg.get('reward_fn_kwargs', {}))

    if env_cfg.get('relabel_env_rewards', False):
        assert learned_rewards == False and reward_fn is not None
        from mbrl.env.wrappers import RelabelEnvRewardsWrapper
        env = RelabelEnvRewardsWrapper(env,
                                       reward_function=reward_fn,
                                       device=cfg.device)

    if cfg.seed is not None:
        env.seed(cfg.seed)
        env.observation_space.seed(cfg.seed + 1)
        env.action_space.seed(cfg.seed + 2)

    return env, term_fn, reward_fn


@hydra.main(config_path="conf", config_name="main_lifelong_learning")
def run(cfg: omegaconf.DictConfig):
    wandb.init(project='fsrl-mbrl',
               config=OmegaConf.to_container(cfg),
               settings=wandb.Settings(start_method="fork"))
    lifelong_learning_envs = []
    lifelong_learning_task_names = []
    lifelong_learning_termination_fns = []
    lifelong_learning_reward_fns = []
    print('[main_lifelong_learning] cfg: ', cfg)

    # num_tasks = len(cfg.overrides.envs)

    for i, env_cfg in enumerate(cfg.overrides.envs):
        print('[main_lifelong_learning:94] wrappers: ', env_cfg.wrappers)
        env, term_fn, reward_fn = make_env(env_cfg.env_name, env_cfg.wrappers,
                                           cfg.overrides.learned_rewards, cfg,
                                           env_cfg)
        # env = MultitaskWrapper(env, i, num_tasks)
        lifelong_learning_envs.append(env)
        lifelong_learning_task_names.append(env_cfg.task_name)
        lifelong_learning_termination_fns.append(term_fn)
        lifelong_learning_reward_fns.append(reward_fn)

    task_name_to_task_index = make_task_name_to_index_map(
        lifelong_learning_task_names)
    num_tasks = len(task_name_to_task_index.keys())
    lifelong_learning_envs = lifelong_learning_envs[:num_tasks]
    lifelong_learning_termination_fns = (
        lifelong_learning_termination_fns[:num_tasks])
    lifelong_learning_reward_fns = lifelong_learning_reward_fns[:num_tasks]
    for i in range(len(lifelong_learning_envs)):
        lifelong_learning_envs[i] = MultitaskWrapper(lifelong_learning_envs[i],
                                                     i, num_tasks)

    print(
        f'[main_lifelong_learning:97] lifelong_learning_envs: {lifelong_learning_envs}'
    )
    print(
        f'[main_lifelong_learning:98] lifelong_learning_termination_fns: {lifelong_learning_termination_fns}'
    )
    print(
        f'[main_lifelong_learning:102] lifelong_learning_reward_fns: {lifelong_learning_reward_fns}'
    )
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return lifelong_learning_pets.train(
            lifelong_learning_envs,
            lifelong_learning_task_names,
            lifelong_learning_termination_fns,
            lifelong_learning_reward_fns,
            lifelong_learning_termination_fns[0],
            cfg,
            forward_postprocess_fn=getattr(lifelong_learning_envs[0],
                                           'forward_postprocess_fn', None))
    # if cfg.algorithm.name == "mbpo":
    #     test_env, *_ = mujoco_util.make_env(cfg)
    #     return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "fsrl":
        fsrl.train(
            lifelong_learning_envs,
            [],  # TODO: ADD EVALUATION ENVIRONMENTS
            lifelong_learning_task_names,
            lifelong_learning_termination_fns,
            lifelong_learning_reward_fns,
            lifelong_learning_termination_fns[0],
            fsrl.PolicyType(cfg.overrides.policy_to_use.upper()),
            cfg,
            forward_postprocess_fn=getattr(lifelong_learning_envs[0],
                                           'forward_postprocess_fn', None))


if __name__ == "__main__":
    run()
