# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple
import gtimer as gt

import gym
import numpy as np
import torch

import mbrl.types
# from mbrl.planning.core import Agent

from . import one_dim_tr_model


class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """
    def __init__(
        self,
        env: gym.Env,
        model: one_dim_tr_model.OneDTransitionRewardModel,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = model.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True

    def reset(
        self,
        initial_obs_batch: np.ndarray,
        return_as_np: bool = True,
    ) -> mbrl.types.TensorType:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (torch.Tensor or np.ndarray): the initial observation in the type indicated
            by ``return_as_np``.
        """
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        batch = mbrl.types.TransitionBatch(
            initial_obs_batch.astype(np.float32), None, None, None, None)
        self._current_obs = self.dynamics_model.reset(batch, rng=self._rng)
        self._return_as_np = return_as_np
        if self._return_as_np:
            return self._current_obs.cpu().numpy()
        return self._current_obs

    @gt.wrap
    def step(
        self,
        actions: mbrl.types.TensorType,
        sample: bool = False,
        mopo_penalty_coeff: float = 0.,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            model_in = mbrl.types.TransitionBatch(self._current_obs, actions,
                                                  None, None, None)
            (next_observs,
             pred_rewards), variances = self.dynamics_model.sample(
                 model_in,
                 deterministic=not sample,
                 rng=self._rng,
                 return_variance=True,
             )
            gt.stamp('sample')
            rewards = (pred_rewards if self.reward_fn is None else
                       self.reward_fn(actions, next_observs))
            if mopo_penalty_coeff != 0.:
                stds = torch.sqrt(variances)
                if len(stds.shape) == 3:
                    penalty, _ = torch.max(torch.norm(stds, dim=-1), dim=0)
                else:
                    penalty = torch.norm(stds, dim=-1)
                penalty = penalty[:, None]
                assert penalty.shape == rewards.shape
                unpenalized_rewards = rewards
                rewards = rewards - mopo_penalty_coeff * penalty
            gt.stamp('reward_function')
            dones = self.termination_fn(actions, next_observs)
            gt.stamp('termination_fn')
            self._current_obs = next_observs
            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, {}

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
        mopo_penalty_coeff: float = 0,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        assert (len(action_sequences.shape) == 3
                )  # population_size, horizon, action_shape
        population_size, horizon, action_dim = action_sequences.shape
        initial_obs_batch = np.tile(
            initial_state,
            (num_particles * population_size, 1)).astype(np.float32)
        self.reset(initial_obs_batch, return_as_np=False)
        batch_size = initial_obs_batch.shape[0]
        total_rewards = torch.zeros(batch_size, 1).to(self.device)
        terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
        for time_step in range(horizon):
            actions_for_step = action_sequences[:, time_step, :]
            action_batch = torch.repeat_interleave(actions_for_step,
                                                   num_particles,
                                                   dim=0)
            _, rewards, dones, _ = self.step(
                action_batch,
                sample=True,
                mopo_penalty_coeff=mopo_penalty_coeff)
            rewards[terminated] = 0
            terminated |= dones
            total_rewards += rewards

        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(dim=1), total_rewards.std(dim=1)

    def evaluate_agent(
        self,
        agent,  #: Agent,
        initial_state: np.ndarray,
        horizon: int,
        num_particles: int,
    ) -> torch.Tensor:
        print('[model_env:189] initial state:', initial_state)
        print('[model_env:190] num_particles:', num_particles)
        initial_obs_batch = np.tile(initial_state,
                                    (num_particles, 1)).astype(np.float32)
        self.reset(initial_obs_batch, return_as_np=False)
        batch_size = initial_obs_batch.shape[0]
        total_rewards = torch.zeros(batch_size, 1).to(self.device)
        terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
        obs = torch.Tensor(initial_obs_batch)
        for time_step in range(horizon):
            actions_for_step = agent.act(obs.to('cpu'), batched=True)
            obs, rewards, dones, _ = self.step(actions_for_step, sample=True)
            rewards[terminated] = 0
            terminated |= dones
            total_rewards += rewards
        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(dim=1), total_rewards.std(dim=1)
