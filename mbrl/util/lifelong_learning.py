import pathlib
from typing import Callable, Dict, List, Tuple, Union, Optional
import omegaconf
import torch
from mbrl.types import RewardFnType
import mbrl.models
from .replay_buffer import (
    BootstrapIterator,
    ReplayBuffer,
    SequenceTransitionIterator,
    TransitionIterator,
    concatenate_batches,
)


def make_task_name_to_index_map(
        lifelong_learning_task_names: List[str]) -> Dict[str, int]:
    """Creates a map going from the task name to the task index.

    The tasks might be repeated, so task indexes aren't just a range.
    """
    task_name_to_task_index = {}
    current_task_index = 0
    for task_name in lifelong_learning_task_names:
        if task_name not in task_name_to_task_index:
            task_name_to_task_index[task_name] = current_task_index
            current_task_index += 1
    return task_name_to_task_index


def separate_observations_and_task_ids(
        original_observations: torch.Tensor,
        num_tasks: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return (original_observations[..., :-num_tasks],
            original_observations[..., -num_tasks:])


def general_reward_function(
    actions: torch.Tensor,
    observations: torch.Tensor,
    list_of_reward_functions: List[RewardFnType],
    device: str,
) -> torch.Tensor:
    obs, task_ids = separate_observations_and_task_ids(
        observations, num_tasks=len(list_of_reward_functions))
    task_ids = task_ids.argmax(dim=-1)
    all_rewards = torch.full([obs.shape[0], 1], -float('inf')).to(device)
    for i, reward_function in enumerate(list_of_reward_functions):
        indices = (task_ids == i).nonzero(as_tuple=True)
        rewards = reward_function(actions[indices], obs[indices], device)
        if (rewards == -float('inf')).sum() != 0:
            print(
                f'[lifelong_learning:52] there is a negative infinity in reward function [{i}]. rewards: {rewards}. rewards.min: {rewards.min()}'
            )
        all_rewards[indices] = rewards
    assert (all_rewards == -float('inf')).sum() == 0
    return all_rewards


def train_lifelong_learning_model_and_save_model_and_data(
    model: mbrl.models.Model,
    model_trainer: mbrl.models.ModelTrainer,
    cfg: omegaconf.DictConfig,
    task_replay_buffers: List[ReplayBuffer],
    work_dir: Optional[Union[str, pathlib.Path]] = None,
    callback: Optional[Callable] = None,
):
    """Convenience function for training a model and saving results.

    Runs `model_trainer.train()`, then saves the resulting model and the data used.
    If the model has an "update_normalizer" method it will be called before training,
    passing `replay_buffer.get_all()` as input.

    Args:
        model (:class:`mbrl.models.Model`): the model to train.
        model_trainer (:class:`mbrl.models.ModelTrainer`): the model trainer.
        cfg (:class:`omegaconf.DictConfig`): configuration to use for training. It
            must contain the following fields::

                -model_batch_size (int)
                -validation_ratio (float)
                -num_epochs_train_model (int, optional)
                -patience (int, optional)
                -bootstrap_permutes (bool, optional)
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer to use.
        work_dir (str or pathlib.Path, optional): if given, a directory to save
            model and buffer to.
        callback (callable, optional): if provided, this function will be called after
            every training epoch. See :class:`mbrl.models.ModelTrainer` for signature.
    """
    data = concatenate_batches([
        replay_buffer.get_all(shuffle=True)
        for replay_buffer in task_replay_buffers
    ])
    dataset_train, dataset_val = mbrl.util.common.get_basic_buffer_iterators(
        None,
        cfg.model_batch_size,
        cfg.validation_ratio,
        ensemble_size=len(model),
        shuffle_each_epoch=True,
        bootstrap_permutes=cfg.get("bootstrap_permutes", False),
        data=data,
        rng=task_replay_buffers[0].rng,
    )
    if hasattr(model, "update_normalizer"):
        model.update_normalizer(data)
    model_trainer.train(
        dataset_train,
        dataset_val=dataset_val,
        num_epochs=cfg.get("num_epochs_train_model", None),
        patience=cfg.get("patience", 1),
        improvement_threshold=cfg.get("improvement_threshold", 0.01),
        callback=callback,
    )
    if work_dir is not None:
        model.save(str(work_dir))
        [
            replay_buffer.save(work_dir, filename_suffix='_task-' + str(i))
            for i, replay_buffer in enumerate(task_replay_buffers)
        ]
