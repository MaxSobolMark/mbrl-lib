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


def metaworld_make_hidden_goal_env(env_name: str, env_cls):
    d = {}

    def initialize(env, seed=None, **kwargs):
        if seed is not None:
            st0 = np.random.get_state()
            np.random.seed(seed)
        super(type(env), env).__init__(**kwargs)
        env._partially_observable = True
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.random_init = False
        env.reset()
        env._freeze_rand_vec = True
        if seed is not None:
            np.random.set_state(st0)

    d['__init__'] = initialize
    hg_env_name = re.sub("(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(),
                         env_name)
    hg_env_name = hg_env_name.replace("-", "")
    hg_env_name = '{}GoalHidden'.format(hg_env_name)
    HiddenGoalEnvCls = type(hg_env_name, (env_cls, ), d)
    return HiddenGoalEnvCls


def make_env(
    env_name: str,
    wrappers: List,
    learned_rewards: bool,
    cfg: omegaconf.DictConfig,
    env_cfg: omegaconf.DictConfig,
) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
    if "gym___" in env_name:
        import mbrl.env

        env = gym.make(env_name.split("___")[1])
        term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
        if hasattr(cfg.overrides,
                   "reward_fn") and cfg.overrides.reward_fn is not None:
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.reward_fn)
        else:
            reward_fn = getattr(mbrl.env.reward_fns, cfg.overrides.term_fn,
                                None)
    elif 'metaworld___' in env_name:
        import metaworld
        import mbrl.env
        env_name = env_name.split('___')[1]
        term_fn = getattr(mbrl.env.termination_fns, cfg.overrides.term_fn)
        no_task = False
        if env_name == 'drawer-open-v2-max':
            from mbrl.env.metaworld_envs import sawyer_drawer_open_v2
            # env = sawyer_drawer_open_v2.SawyerDrawerOpenEnvV2(
            #     **env_cfg.env_kwargs)
            # mt1 = metaworld.MT1('drawer-open-v2')
            # from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
            # drawer_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[
            #     'drawer-open-v2-goal-hidden']
            # env = drawer_open_goal_hidden_cls()
            env = metaworld_make_hidden_goal_env(
                env_name, sawyer_drawer_open_v2.SawyerDrawerOpenEnvV2)(
                    **env_cfg.env_kwargs)
            no_task = True
        elif env_name == 'drawer-close-v2-max':
            from mbrl.env.metaworld_envs import sawyer_drawer_close_v2
            # env = sawyer_drawer_close_v2.SawyerDrawerCloseEnvV2(
            #     **env_cfg.env_kwargs)
            # mt1 = metaworld.MT1('drawer-close-v2')
            # from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
            # drawer_close_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[
            #     'drawer-close-v2-goal-hidden']
            # env = drawer_close_goal_hidden_cls()
            env = metaworld_make_hidden_goal_env(
                env_name, sawyer_drawer_close_v2.SawyerDrawerCloseEnvV2)(
                    **env_cfg.env_kwargs)
            no_task = True
        elif env_name == 'faucet-open-v2-max':
            # from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
            # faucet_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[
            #     'faucet-open-v2-goal-hidden']
            # env = faucet_open_goal_hidden_cls()
            from mbrl.env.metaworld_envs import sawyer_faucet_open_v2
            env = metaworld_make_hidden_goal_env(
                env_name, sawyer_faucet_open_v2.SawyerFaucetOpenEnvV2)(
                    **env_cfg.env_kwargs)
            no_task = True
        elif env_name == 'faucet-close-v2-max':
            # from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
            # faucet_close_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[
            #     'faucet-close-v2-goal-hidden']
            # env = faucet_close_goal_hidden_cls()
            from mbrl.env.metaworld_envs import sawyer_faucet_close_v2
            env = metaworld_make_hidden_goal_env(
                env_name, sawyer_faucet_close_v2.SawyerFaucetCloseEnvV2)(
                    **env_cfg.env_kwargs)
            no_task = True
        else:
            mt1 = metaworld.MT1(env_name)
            env = mt1.train_classes[list(mt1.train_classes)[0]]()
        if not no_task:
            task_index = env_cfg.task_index
            task = mt1.train_tasks[task_index]
            env.set_task(task)
        reward_fn = DOMAIN_TO_REWARD_FUNCTION[env_name](env)
        env = gym.wrappers.TimeLimit(env,
                                     max_episode_steps=cfg.overrides.get(
                                         "trial_length", 500))
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
            env = mbrl.env.mujoco_envs.AntTruncatedObsEnv()
            term_fn = mbrl.env.termination_fns.ant
            reward_fn = None
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
        print('[main_lifelong_learning:155] i: ', str(i),
              '; env.observation_space: ', env.observation_space)
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
        lifelong_learning_envs[i] = MultitaskWrapper(
            lifelong_learning_envs[i],
            i,
            num_tasks,
            include_task_id_in_obs_space=cfg.overrides.observe_task_id)

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
        fsrl.train(lifelong_learning_envs,
                   lifelong_learning_envs,
                   lifelong_learning_task_names,
                   lifelong_learning_termination_fns,
                   lifelong_learning_reward_fns,
                   lifelong_learning_termination_fns[0],
                   fsrl.PolicyType(cfg.overrides.policy_to_use.upper()),
                   cfg,
                   forward_postprocess_fn=getattr(lifelong_learning_envs[0],
                                                  'forward_postprocess_fn',
                                                  None))


if __name__ == "__main__":
    run()
