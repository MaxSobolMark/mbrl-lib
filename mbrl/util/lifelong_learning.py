from typing import Dict, List, Tuple
import torch


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
