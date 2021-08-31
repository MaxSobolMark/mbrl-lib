# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Callable, Optional, List, Tuple
from enum import Enum
import gtimer as gt
from functools import partial

import gym
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
from mbrl.models.lifelong_learning_model import LifelongLearningModel
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
from mbrl.util.lifelong_learning import (
    make_task_name_to_index_map, general_reward_function,
    train_lifelong_learning_model_and_save_model_and_data)
import mbrl.util.math

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT

TIMES_LOG_FORMAT = [
    ('initialization', 'i', 'float'),
    ('train_model', 't', 'float'),
    ('step_env', 's_e', 'float'),
    ('finished_step', 'f_s', 'float'),
    ('finished_task', 'f_t', 'float'),
]


class PolicyType(Enum):
    EXPLICIT_POLICY = 'EXPLICIT_POLICY'
    CEM_PLANNING = 'CEM_PLANNING'
    COMBINED = 'COMBINED'

    def __str__(self) -> str:
        return self.value


def train(
    lifelong_learning_envs: List[gym.Env],
    lifelong_learning_task_names: List[str],
    lifelong_learning_termination_fns: List[mbrl.types.TermFnType],
    lifelong_learning_reward_fns: List[mbrl.types.RewardFnType],
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
    forward_postprocess_fn: Callable[[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.parameter.Parameter
    ], Tuple[torch.Tensor, torch.Tensor]] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = lifelong_learning_envs[0].observation_space.shape
    act_shape = lifelong_learning_envs[0].action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(mbrl.constants.RESULTS_LOG_NAME,
                              EVAL_LOG_FORMAT,
                              color="green")
        logger.register_group('times', TIMES_LOG_FORMAT)
    gt.reset_root()
    gt.rename_root('lifelong_learning_pets')
    gt.set_def_unique(False)

    # -------- Create and populate initial env dataset --------
    task_name_to_task_index = make_task_name_to_index_map(
        lifelong_learning_task_names)
    num_tasks = len(task_name_to_task_index.keys())
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(
        cfg, obs_shape, act_shape)
    # print('[lifelong_learning_pets:78] forw_postproc: ',
    #       cfg.overrides.forward_postprocess_fn)
    # if cfg.overrides.forward_postprocess_fn != 'None':
    #     forward_postprocess_fn = cfg.overrides.forward_postprocess_fn
    # else:
    #     forward_postprocess_fn = None
    dynamics_model = LifelongLearningModel(
        dynamics_model,
        num_tasks,
        obs_shape,
        act_shape,
        observe_task_id=cfg.overrides.observe_task_id,
        forward_postprocess_fn=forward_postprocess_fn,
    )
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    task_replay_buffers = [
        mbrl.util.common.create_replay_buffer(
            cfg,
            obs_shape,
            act_shape,
            rng=rng,
            obs_type=dtype,
            action_type=dtype,
            reward_type=dtype,
        ) for _ in range(num_tasks)
    ]
    mbrl.util.common.rollout_agent_trajectories(
        lifelong_learning_envs[0],
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(lifelong_learning_envs[0]),
        {},
        replay_buffer=task_replay_buffers[0],
    )
    task_replay_buffers[0].save(work_dir)

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    reward_function = partial(
        general_reward_function,
        list_of_reward_functions=lifelong_learning_reward_fns,
        device=cfg.device)
    print('[lifelong_learning_pets:135] lifelong_learning_reward_fns: ',
          lifelong_learning_reward_fns)
    model_env = mbrl.models.ModelEnv(lifelong_learning_envs[0],
                                     dynamics_model,
                                     termination_fn,
                                     reward_function,
                                     generator=torch_generator)
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env,
        cfg.algorithm.agent,
        num_particles=cfg.algorithm.num_particles)
    gt.stamp('initialization')

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    # loop = gt.timed_loop('training_loop')
    # while env_steps < cfg.overrides.num_steps:
    # next(loop)
    # task_loop = gt.timed_loop('task_loop')
    for i in range(len(lifelong_learning_task_names)):
        env_steps_this_task = 0
        while env_steps_this_task < cfg.overrides.num_steps // len(
                lifelong_learning_task_names):

            # next(task_loop)
            task_i = task_name_to_task_index[lifelong_learning_task_names[i]]
            obs = lifelong_learning_envs[task_i].reset()
            agent.reset()
            done = False
            total_reward = 0.0
            steps_trial = 0
            #episode_loop = gt.timed_loop('episode_loop')
            while not done:
                # next(episode_loop)
                # --------------- Model Training -----------------
                if env_steps % cfg.algorithm.freq_train_model == 0:
                    train_lifelong_learning_model_and_save_model_and_data(
                        dynamics_model,
                        model_trainer,
                        cfg.overrides,
                        task_replay_buffers,
                        work_dir=work_dir,
                    )
                    gt.stamp('train_model')

                # --- Doing env step using the agent and adding to model dataset ---
                print('[lifelong_learning_pets:189] task_i: ', task_i)
                print('[lifelong_learning_pets:190] obs: ', obs)
                model_env.reset(np.tile(np.reshape(obs, [1, -1]), [5, 1]),
                                return_as_np=False)
                imagined_obs, _, _, _ = model_env.step(np.tile(
                    np.reshape(
                        lifelong_learning_envs[task_i].action_space.sample(),
                        [1, -1]), [5, 1]),
                                                       sample=False)
                print('[lifelong_learning_pets:195] imagined_obs: ',
                      imagined_obs[0])
                next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                    lifelong_learning_envs[task_i], obs, agent, {},
                    task_replay_buffers[task_i])
                gt.stamp('step_env')

                obs = next_obs
                total_reward += reward
                steps_trial += 1
                env_steps += 1
                env_steps_this_task += 1

                if debug_mode:
                    print(f"Step {env_steps}: Reward {reward:.3f}.")
                    # print(gt.report())
                # gt.stamp('episode_loop')

            #episode_loop.exit()

            if logger is not None:
                time_diagnostics = {
                    key: times[-1]
                    for key, times in gt.get_times().stamps.itrs.items()
                }
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "env_step": env_steps,
                        "episode_reward": total_reward,
                        "task_index": task_i,
                    },
                )
                logger.log_data('times', time_diagnostics)
                print(gt.report(include_itrs=False))
            current_trial += 1
            if debug_mode:
                print(f"Trial: {current_trial }, reward: {total_reward}.")

            max_total_reward = max(max_total_reward, total_reward)
            gt.stamp('finished_step')
        # task_loop.exit()
        gt.stamp('finished_task')
    # loop.exit()
    print(gt.report(include_itrs=False))
    return np.float32(max_total_reward)
