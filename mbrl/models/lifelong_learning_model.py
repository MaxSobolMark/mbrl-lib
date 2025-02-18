# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import pickle
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import hydra
import gtimer as gt

import numpy as np
import omegaconf
import torch

import mbrl.models.util as model_util
import mbrl.types
import mbrl.util.math
from mbrl.util.lifelong_learning import separate_observations_and_task_ids

from .model import Ensemble, LossOutput, Model, UpdateOutput
from mbrl.types import ModelInput
from torch.nn import functional as F

MODEL_LOG_FORMAT = [
    ("train_iteration", "I", "int"),
    ("epoch", "E", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_score", "MSCORE", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]


class LifelongLearningModel():  # Model):
    """Wrapper class for 1-D dynamics models.
    """

    _MODEL_FNAME = "model.pth"
    _ELITE_FNAME = "elite_models.pkl"

    def __init__(
        self,
        model: Model,
        num_tasks: int,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        cfg: omegaconf.DictConfig,
        observe_task_id: bool = False,
        forward_postprocess_fn: Callable[[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.parameter.
            Parameter
        ], Tuple[torch.Tensor, torch.Tensor]] = lambda inputs, mean, logvar:
        (mean, logvar),
    ):
        super().__init__()
        self._model = model
        self._num_tasks = num_tasks
        self._obs_shape = obs_shape
        self._act_shape = act_shape
        self._cfg = cfg
        self._observe_task_id = observe_task_id
        self._forward_postprocess_fn = forward_postprocess_fn
        self.device = model.device
        self._original_forward = self._model.model.forward
        self._model.model.forward = self.forward
        # Make the dimensions of the task ID not delta.
        # Extend the no_delta_list with -1, -2, -3, ..., -self._num_tasks.
        print('self._model.no_delta_list: ', self._model.no_delta_list)
        self._model.no_delta_list.extend(
            list(range(-1, -self._num_tasks - 1, -1)))

    @gt.wrap
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Calls forward method of base model with the given input and args."""
        original_inputs = x
        # if not self._observe_task_id:
        # x contains the observations and the actions
        observations = x[..., :-np.prod(self._act_shape)]
        # if self._num_tasks > 1:
        observations, task_ids = separate_observations_and_task_ids(
            observations, self._num_tasks)
        if self._num_tasks > 1:
            assert task_ids.min(
            ) == 0., f'task_ids.min(): {task_ids.min()}\ntask_ids: {task_ids}'
            assert task_ids.max() == 1., f'task_ids.max(): {task_ids.max()}'
        assert torch.all(
            torch.logical_or(
                task_ids.eq(torch.ones_like(task_ids).to(self.device)),
                task_ids.eq(torch.zeros_like(task_ids).to(self.device))))
        if self._observe_task_id:
            observations = torch.cat([observations, task_ids], dim=-1)
        x = torch.cat([observations, x[..., -np.prod(self._act_shape):]],
                      dim=-1)
        # x[..., :-np.prod(self._act_shape)] = observations
        gt.stamp('forward_preprocessing')
        mean, logvar = self._original_forward(x, **kwargs)
        gt.stamp('original_forward')
        # if self._num_tasks > 1:
        if not self._observe_task_id:
            task_ids_for_concatenation = torch.broadcast_to(
                task_ids, mean.shape[:-1] + (task_ids.shape[-1], ))
            if not self._cfg.overrides.learned_rewards:
                mean = torch.cat([mean, task_ids_for_concatenation], dim=-1)
                logvar = torch.cat([logvar, task_ids_for_concatenation],
                                   dim=-1)
            else:
                mean = torch.cat([
                    mean[..., :-1], task_ids_for_concatenation,
                    mean[..., -1][..., None]
                ],
                                 dim=-1)
                logvar = torch.cat([
                    logvar[..., :-1], task_ids_for_concatenation,
                    logvar[..., -1][..., None]
                ],
                                   dim=-1)
        if not self._cfg.overrides.learned_rewards:
            mean[..., -self._num_tasks:] = task_ids.detach()
            logvar[..., -self._num_tasks:] = (torch.ones_like(task_ids) *
                                              -float('inf')).detach()
        else:
            mean[..., -(self._num_tasks + 1):-1] = task_ids.detach()
            logvar[...,
                   -(self._num_tasks + 1):-1] = (torch.ones_like(task_ids) *
                                                 -float('inf')).detach()
        if self._forward_postprocess_fn is not None:
            mean, logvar = self._forward_postprocess_fn(
                original_inputs, mean, logvar, self._model.model.min_logvar)
        gt.stamp('forward_postprocessing')

        return mean, logvar

    def __len__(self, *args, **kwargs):
        return self._model.__len__(*args, **kwargs)

    # def loss(self, *args, **kwargs):
    #     return self._model.loss(*args, **kwargs)

    def eval_score(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert target is None
        with torch.no_grad():
            model_in, target = self._get_model_input_and_target_from_batch(
                batch)
            assert model_in.ndim == 2 and target.ndim == 2
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self._model.model.num_members, 1, 1))
            return F.mse_loss(pred_mean, target, reduction="none"), {}

    # def update_normalizer(self, *args, **kwargs):
    #     return self._model.update_normalizer(*args, **kwargs)

    def update(
        self,
        batch: mbrl.types.TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ):
        assert target is None
        model_in, target = self._get_model_input_and_target_from_batch(batch)
        return self.model.update(model_in, optimizer, target=target)

    # def get_output_and_targets(self, *args, **kwargs):
    #     return self._model.get_output_and_targets(*args, **kwargs)

    # def sample(self, *args, **kwargs):
    #     return self._model.sample(*args, **kwargs)

    # def reset(self, *args, **kwargs):
    #     return self._model.reset(*args, **kwargs)

    # def save(self, *args, **kwargs):

    def __getattr__(self, attr_name):
        return getattr(self._model, attr_name)
