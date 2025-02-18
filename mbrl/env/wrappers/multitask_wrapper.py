"""Multitask wrapper that appends to observations a one-hot vector with the
   task index."""

from gym import ObservationWrapper, spaces
import numpy as np


class MultitaskWrapper(ObservationWrapper):
    def __init__(self,
                 env,
                 task_index: int,
                 number_of_tasks: int,
                 include_task_id_in_obs_space: bool = True):
        super(MultitaskWrapper, self).__init__(env)
        # if include_task_id_in_obs_space:
        if True:
            low = np.concatenate(
                [env.observation_space.low, [0] * number_of_tasks], axis=-1)
            high = np.concatenate(
                [env.observation_space.high, [1] * number_of_tasks], axis=-1)
        else:
            low = env.observation_space.low
            high = env.observation_space.high
        self.observation_space = spaces.Box(low, high)
        self._task_index = task_index
        self._number_of_tasks = number_of_tasks

    def observation(self, observation):
        obs = np.concatenate(
            [observation,
             np.eye(self._number_of_tasks)[self._task_index]],
            axis=-1)
        return obs

    def __str__(self):
        return f'MultitaskWrapper<{str(self.env)}>(task {self._task_index}_of_{self._number_of_tasks})'
