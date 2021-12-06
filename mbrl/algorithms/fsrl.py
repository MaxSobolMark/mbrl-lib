# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Callable, Optional, List, Tuple
from enum import Enum
import gtimer as gt
from functools import partial
import hydra.utils

import gym
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
from mbrl.models.lifelong_learning_model import LifelongLearningModel
import mbrl.planning
import mbrl.third_party.pytorch_sac as pytorch_sac
import mbrl.types
import mbrl.util
import mbrl.util.common
from mbrl.util.lifelong_learning import (
    make_task_name_to_index_map, general_reward_function,
    train_lifelong_learning_model_and_save_model_and_data)
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent

from .mbpo import (MBPO_LOG_FORMAT, rollout_model_and_populate_sac_buffer,
                   evaluate, maybe_replace_sac_buffer)

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT
POLICY_DIAGNOSTICS_FORMAT = [
    ('explicit_policy_returns_expectation', 'EP', 'float'),
    ('planning_policy_returns_expectation', 'PP', 'float'),
]


class PolicyType(Enum):
    EXPLICIT_POLICY = 'EXPLICIT_POLICY'
    CEM_PLANNING = 'CEM_PLANNING'
    COMBINED = 'COMBINED'

    def __str__(self) -> str:
        return self.value


def train(
    lifelong_learning_envs: List[gym.Env],
    evaluation_environments: List[gym.Env],
    lifelong_learning_task_names: List[str],
    lifelong_learning_termination_fns: List[mbrl.types.TermFnType],
    lifelong_learning_reward_fns: List[mbrl.types.RewardFnType],
    termination_fn: mbrl.types.TermFnType,
    policy_to_use: PolicyType,
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
        logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
        logger.register_group(mbrl.constants.RESULTS_LOG_NAME,
                              EVAL_LOG_FORMAT,
                              color="green")
        logger.register_group('policy_diagnostics',
                              POLICY_DIAGNOSTICS_FORMAT,
                              color='red')
    gt.reset_root()
    gt.rename_root('lifelong_learning_pets')
    gt.set_def_unique(False)

    # MBPO specific initialization
    mbrl.planning.complete_agent_cfg(lifelong_learning_envs[0],
                                     cfg.algorithm.sac_agent)
    sac_agent = hydra.utils.instantiate(cfg.algorithm.sac_agent)

    save_video = cfg.get("save_video", False)
    video_recorder = pytorch_sac.VideoRecorder(
        work_dir if save_video else None)

    # -------- Create and populate initial env dataset --------
    task_name_to_task_index = make_task_name_to_index_map(
        lifelong_learning_task_names)
    num_tasks = len(task_name_to_task_index.keys())
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(
        cfg, obs_shape, act_shape)
    print('[lifelong_learning_pets:78] forw_postproc: ',
          cfg.overrides.forward_postprocess_fn)
    # if cfg.overrides.forward_postprocess_fn != 'None':
    #     forward_postprocess_fn = cfg.overrides.forward_postprocess_fn
    # else:
    #     forward_postprocess_fn = None
    dynamics_model = LifelongLearningModel(
        dynamics_model,
        num_tasks,
        obs_shape,
        act_shape,
        cfg,
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
    # Keep a set of the task pools that have experiences on them.
    active_task_buffers = set({task_replay_buffers[0]})
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        lifelong_learning_envs[0],
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(lifelong_learning_envs[0])
        if random_explore else sac_agent,
        {} if random_explore else {
            "sample": True,
            "batched": False
        },
        replay_buffer=task_replay_buffers[0],
    )
    task_replay_buffers[0].save(work_dir)

    # ---------------------------------------------------------
    # ---------- Create model environment and planning agent --
    reward_function = partial(
        general_reward_function,
        list_of_reward_functions=lifelong_learning_reward_fns,
        device=cfg.device)
    model_env = mbrl.models.ModelEnv(lifelong_learning_envs[0],
                                     dynamics_model,
                                     termination_fn,
                                     reward_function,
                                     generator=torch_generator)
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )

    planning_agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env,
        cfg.algorithm.agent,
        num_particles=cfg.algorithm.num_particles,
        planning_mopo_penalty_coeff=cfg.algorithm.planning_mopo_penalty_coeff)
    steps_trial = 0
    env_steps_this_task = 0
    if policy_to_use == PolicyType.CEM_PLANNING:
        agent = planning_agent
    elif policy_to_use == PolicyType.EXPLICIT_POLICY:
        agent = sac_agent
    elif policy_to_use == PolicyType.COMBINED:
        agent = mbrl.planning.CombinedAgent(sac_agent, planning_agent,
                                            model_env, cfg)
        original_act = agent.act

        def act_wrapper(obs, **kwargs):
            return original_act(
                obs,
                current_task_index=task_i,
                timestep_in_epoch=steps_trial,
                current_task_replay_buffer=task_replay_buffers[task_i],
                env_steps_this_task=env_steps_this_task,
                **kwargs)

        agent.act = act_wrapper
    else:
        raise NotImplementedError

    gt.stamp('initialization')

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    mbpo_rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step *
        cfg.algorithm.freq_train_model)
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model))
    updates_made = 0
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    best_eval_reward = -np.Inf
    epoch = 0
    sac_buffer = None
    # while env_steps < cfg.overrides.num_steps:
    #     for i in range(cfg.overrides.num_steps //
    #                    len(lifelong_learning_task_names)):
    for i in range(len(lifelong_learning_task_names)):
        env_steps_this_task = 0
        task_i = task_name_to_task_index[lifelong_learning_task_names[i]]
        active_task_buffers.add(task_replay_buffers[task_i])
        while env_steps_this_task < cfg.overrides.num_steps // len(
                lifelong_learning_task_names):
            # next(task_loop)
            obs = lifelong_learning_envs[task_i].reset()
            agent.reset()
            done = False
            total_reward = 0.0
            steps_trial = 0
            # MBPO params
            rollout_length = int(
                mbrl.util.math.truncated_linear(
                    *(cfg.overrides.rollout_schedule + [epoch + 1])))
            sac_buffer_capacity = (rollout_length * mbpo_rollout_batch_size *
                                   trains_per_epoch)
            sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
            sac_buffer = maybe_replace_sac_buffer(
                sac_buffer,
                sac_buffer_capacity,
                obs_shape,
                act_shape,
                torch.device(cfg.device),
            )
            while not done:
                # --------------- Model Training -----------------
                if env_steps % cfg.algorithm.freq_train_model == 0:
                    train_lifelong_learning_model_and_save_model_and_data(
                        dynamics_model,
                        model_trainer,
                        cfg.overrides,
                        task_replay_buffers,
                        work_dir=work_dir,
                        agent=agent,
                    )
                    gt.stamp('train_model')

                # --- Doing env step using the agent and adding to model dataset ---
                next_obs, reward, done, step_info = mbrl.util.common.step_env_and_add_to_buffer(
                    lifelong_learning_envs[task_i], obs, agent, {},
                    task_replay_buffers[task_i])
                gt.stamp('step_env')

                # Rollout model and store imagined trajectories for MBPO.
                # TODO!!!!: make replay buffer sample equally from all tasks.
                num_active_buffers = len(active_task_buffers)
                real_batches_for_rollout = [
                    replay_buffer.sample(mbpo_rollout_batch_size //
                                         num_active_buffers)
                    for replay_buffer in active_task_buffers
                ]
                # mbpo_rollout_batch_size might not be divisible by the active buffers
                # so get remaining samples
                if mbpo_rollout_batch_size % num_active_buffers != 0:
                    real_batches_for_rollout.append(
                        task_replay_buffers[task_i].sample(
                            mbpo_rollout_batch_size -
                            mbpo_rollout_batch_size // num_active_buffers *
                            num_active_buffers))
                real_batches_for_rollout = (
                    mbrl.util.replay_buffer.concatenate_batches(
                        real_batches_for_rollout))
                if rollout_length > 0:
                    if hasattr(agent, 'should_mopo_for_policy_be_used'):
                        policy_mopo_penalty_coeff = (
                            cfg.overrides.policy_mopo_penalty_coeff
                            if agent.should_mopo_for_policy_be_used() else 0.)
                    else:
                        policy_mopo_penalty_coeff = (
                            cfg.overrides.policy_mopo_penalty_coeff)
                    rollout_model_and_populate_sac_buffer(
                        model_env,
                        None,
                        sac_agent,
                        sac_buffer,
                        cfg.algorithm.sac_samples_action,
                        rollout_length,
                        mbpo_rollout_batch_size,
                        batch=real_batches_for_rollout,
                        mopo_penalty_coeff=policy_mopo_penalty_coeff)
                if debug_mode:
                    print(f"Epoch: {epoch}. "
                          f"SAC buffer size: {len(sac_buffer)}. "
                          f"Rollout length: {rollout_length}. "
                          f"Steps: {env_steps}")
                # SAC Agent Training
                for _ in range(cfg.overrides.num_sac_updates_per_step):
                    if ((env_steps + 1) % cfg.overrides.sac_updates_every_steps
                            != 0 or len(sac_buffer) < mbpo_rollout_batch_size):
                        break  # only update every once in a while
                    sac_agent.update(sac_buffer, logger, updates_made)
                    updates_made += 1
                    if (not silent
                            and updates_made % cfg.log_frequency_agent == 0):
                        logger.dump(updates_made, save=True)

                obs = next_obs
                total_reward += reward
                steps_trial += 1
                env_steps += 1
                env_steps_this_task += 1

                if debug_mode:
                    print(f"Step {env_steps}: Reward {reward:.3f}.")

            # Episode finished
            # TODO: add support for early termination.
            if logger is not None:
                if policy_to_use == PolicyType.COMBINED:
                    active_policy = agent.get_active_policy()
                elif policy_to_use == PolicyType.CEM_PLANNING:
                    active_policy = 1
                else:
                    active_policy = 0
                results_dict = {
                    "epoch": epoch,
                    "env_step": env_steps,
                    "episode_reward": total_reward,
                    "rollout_length": rollout_length,
                    "task_index": task_i,
                    'active_policy': active_policy,
                }
                results_dict.update(step_info)
                results_dict.pop('TimeLimit.truncated', None)
                logger.log_data(mbrl.constants.RESULTS_LOG_NAME, results_dict)
                if hasattr(agent, 'get_episode_diagnostics'):
                    logger.log_data('policy_diagnostics',
                                    agent.get_episode_diagnostics())
            for i, test_env in enumerate(evaluation_environments):
                evaluate(test_env, sac_agent, cfg.algorithm.num_eval_episodes,
                         video_recorder)
                video_recorder.save(f"{current_trial}_task_{i}.mp4")
            torch.save(sac_agent.critic.state_dict(),
                       os.path.join(work_dir, "critic.pth"))
            torch.save(sac_agent.actor.state_dict(),
                       os.path.join(work_dir, "actor.pth"))
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
