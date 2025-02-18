import numpy as np
import torch
from typing import Tuple, Dict

import omegaconf
from mbrl.util.replay_buffer import ReplayBuffer
import mbrl.models
from .core import Agent
from .trajectory_opt import TrajectoryOptimizerAgent
from .sac_wrapper import SACAgent


class CombinedAgent(Agent):
    """Agent that imagines ahead to choose either tha planning agent or the
       explicit policy agent.
    """
    def __init__(
        self,
        sac_agent: SACAgent,
        planning_agent: TrajectoryOptimizerAgent,
        model_env: mbrl.models.ModelEnv,
        cfg: omegaconf.DictConfig,
    ):
        self.sac_agent = sac_agent
        self.planning_agent = planning_agent
        self._model_env = model_env
        self._cfg = cfg
        self._policy_to_use_current_epoch = None
        self._diagnostics = {}

    def get_active_policy(self):
        return 1 if self._policy_to_use_current_epoch == 'planner' else 0

    def act(self,
            obs: np.ndarray,
            current_task_index: int,
            timestep_in_epoch: int,
            current_task_replay_buffer: ReplayBuffer,
            env_steps_this_task: int,
            return_plan: bool = False,
            **kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation (or batch of observations) for which the action
                is needed.
            sample (bool): if ``True`` the agent samples actions from its policy, otherwise it
                returns the mean policy value. Defaults to ``False``.
            batched (bool): if ``True`` signals to the agent that the obs should be interpreted
                as a batch.

        Returns:
            (np.ndarray): the action.
        """
        if timestep_in_epoch == 0:
            print('[combined_agent:55] current_task_index: ',
                  current_task_index)
            if self._cfg.overrides.get('naive_switching', False):
                if env_steps_this_task < self._cfg.overrides.trial_length * 3:
                    self._policy_to_use_current_epoch = 'planner'
                else:
                    self._policy_to_use_current_epoch = 'explicit_policy'
            else:
                returns_expectation_calculation_method = (
                    self._cfg.overrides.returns_expectation_calculation_method)
                if returns_expectation_calculation_method == 'last_n_episodes':
                    last_n_trajectories = (
                        current_task_replay_buffer.get_last_n_trajectories(
                            self._cfg.overrides.returns_expectation_n_episodes)
                    )
                    if last_n_trajectories is not None:
                        obs, *_ = last_n_trajectories.astuple()
                # task_episodes = current_task_replay_buffer.get_all()
                # TODO: DONT EVALUATE WITH EVERYTHING IN THE REPLAY BUFFER
                planner_action_sequence = self.planning_agent.plan(obs)[
                    np.newaxis]
                print('[combined_agent:49] planner_action_sequence.shape: ',
                      planner_action_sequence.shape)
                planner_returns_expectation, planner_returns_std = (
                    self._model_env.evaluate_action_sequences(
                        torch.Tensor(planner_action_sequence).to(
                            self._cfg.device),
                        initial_state=obs,
                        num_particles=self._cfg.algorithm.num_particles))
                (explicit_policy_returns_expectation,
                 explicit_policy_returns_std) = self._model_env.evaluate_agent(
                     self.sac_agent, obs, self._cfg.overrides.planning_horizon,
                     self._cfg.algorithm.num_particles)

                if self._cfg.get('combined_policy_prefer_explicit_policy',
                                 False):
                    self._policy_to_use_current_epoch = (
                        'planner' if planner_returns_expectation >
                        explicit_policy_returns_expectation +
                        explicit_policy_returns_std else 'explicit_policy')
                else:
                    self._policy_to_use_current_epoch = (
                        'planner' if planner_returns_expectation >
                        explicit_policy_returns_expectation else
                        'explicit_policy')
                self._diagnostics = {
                    'planner_returns_expectation':
                    planner_returns_expectation,
                    'explicit_policy_returns_expectation':
                    explicit_policy_returns_expectation,
                }

        if self._policy_to_use_current_epoch == 'planner':
            if return_plan:
                return self.planning_agent.plan(obs, **kwargs)
            return self.planning_agent.act(obs, **kwargs)
        else:
            if return_plan:
                return self.sac_agent.plan(obs, **kwargs)
            return self.sac_agent.act(obs, **kwargs)

        # with pytorch_sac_utils.eval_mode(), torch.no_grad():
        # return self.sac_agent.act(obs, sample=sample, batched=batched)

    def plan(self, *args, **kwargs) -> np.ndarray:
        return self.act(*args, return_plan=True, **kwargs)

    def reset(self):
        self.sac_agent.reset()
        self.planning_agent.reset()
        self._policy_to_use_current_epoch = None

    def get_episode_diagnostics(self) -> Dict[str, float]:
        return self._diagnostics
