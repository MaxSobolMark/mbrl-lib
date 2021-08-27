# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, List
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
from mbrl.util.lifelong_learning import (make_task_name_to_index_map,
                                         general_reward_function)
import mbrl.util.math

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT


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

    # -------- Create and populate initial env dataset --------
    task_name_to_task_index = make_task_name_to_index_map(
        lifelong_learning_task_names)
    num_tasks = len(task_name_to_task_index.keys())
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(
        cfg, obs_shape, act_shape)
    dynamics_model = LifelongLearningModel(
        dynamics_model,
        num_tasks,
        observe_task_id=cfg.overrides.lifelong_learning.observe_task_id,
        forward_postprocess_fn=cfg.overrides.lifelong_learning.
        forward_postprocess_fn,
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
    reward_function = partial(general_reward_function, num_tasks=num_tasks)
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

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    while env_steps < cfg.overrides.num_steps:
        for i in range(cfg.overrides.num_steps //
                       lifelong_learning_task_names):
            task_i = task_name_to_task_index[lifelong_learning_task_names[i]]
            obs = lifelong_learning_envs[task_i].reset()
            agent.reset()
            done = False
            total_reward = 0.0
            steps_trial = 0
            while not done:
                # --------------- Model Training -----------------
                if env_steps % cfg.algorithm.freq_train_model == 0:
                    mbrl.util.common.train_lifelong_learning_model_and_save_model_and_data(
                        dynamics_model,
                        model_trainer,
                        cfg.overrides,
                        task_replay_buffers,
                        work_dir=work_dir,
                    )

                # --- Doing env step using the agent and adding to model dataset ---
                next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                    lifelong_learning_envs[task_i], obs, agent, {},
                    task_replay_buffers[task_i])

                obs = next_obs
                total_reward += reward
                steps_trial += 1
                env_steps += 1

                if debug_mode:
                    print(f"Step {env_steps}: Reward {reward:.3f}.")

            if logger is not None:
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "env_step": env_steps,
                        "episode_reward": total_reward
                    },
                )
            current_trial += 1
            if debug_mode:
                print(f"Trial: {current_trial }, reward: {total_reward}.")

            max_total_reward = max(max_total_reward, total_reward)

    return np.float32(max_total_reward)
