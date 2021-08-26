# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import pickle
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import hydra

import numpy as np
import torch

import mbrl.models.util as model_util
import mbrl.types
import mbrl.util.math
from mbrl.util.lifelong_learning import separate_observations_and_task_ids

from .model import Ensemble, LossOutput, Model, UpdateOutput

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
        observe_task_id: bool = False,
        forward_postprocess_fn: Callable[[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.parameter.
            Parameter
        ], Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self._model = model
        self._num_tasks = num_tasks
        self._observe_task_id = observe_task_id
        self._forward_postprocess_fn = forward_postprocess_fn
        self.device = model.device
        self._original_forward = self._model.model.forward
        self._model.model.forward = self.forward

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Calls forward method of base model with the given input and args."""
        original_inputs = x
        # if not self._observe_task_id:
        if self._num_tasks > 1:
            x, task_ids = separate_observations_and_task_ids(
                x, self._num_tasks)
            assert torch.equal(task_ids.min(), torch.zeros(1).to(self.device))
            assert torch.equal(task_ids.max(), torch.ones(1).to(self.device))
            assert torch.logical_or(
                task_ids.eq(torch.ones_like(task_ids).to(self.device)),
                task_ids.eq(torch.zeros_like(task_ids).to(self.device)))
            if self._observe_task_id:
                x = torch.cat([x, task_ids], dim=-1)
        mean, logvar = self._original_forward(x, **kwargs)
        if self._num_tasks > 1:
            mean[..., -self._num_tasks:] = task_ids.detach()
            logvar[..., -self._num_tasks:] = (
                torch.ones_like(task_ids) *
                self._model.model.min_logvar).detach()

        mean, logvar = self._forward_postprocess_fn(
            original_inputs, mean, logvar, self._model.model.min_logvar)
        return mean, logvar

    def __len__(self, *args, **kwargs):
        return self._model.__len__(*args, **kwargs)

    # def loss(self, *args, **kwargs):
    #     return self._model.loss(*args, **kwargs)

    # def eval_score(self, *args, **kwargs):
    #     return self._model.eval_score(*args, **kwargs)

    # def update_normalizer(self, *args, **kwargs):
    #     return self._model.update_normalizer(*args, **kwargs)

    # def update(self, *args, **kwargs):
    #     return self._model.update(*args, **kwargs)

    # def get_output_and_targets(self, *args, **kwargs):
    #     return self._model.get_output_and_targets(*args, **kwargs)

    # def sample(self, *args, **kwargs):
    #     return self._model.sample(*args, **kwargs)

    # def reset(self, *args, **kwargs):
    #     return self._model.reset(*args, **kwargs)

    # def save(self, *args, **kwargs):

    def __getattr__(self, attr_name):
        return getattr(self._model, attr_name)
