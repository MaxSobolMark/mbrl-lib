from abc import ABCMeta, abstractmethod
import torch


class BaseRewardFunction(metaclass=ABCMeta):
    def __init__(self, environment):
        self._env = environment

    @property
    @abstractmethod
    def OBS_DIM(self):
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, observation: torch.Tensor, action: torch.Tensor,
                   device: str):
        raise NotImplementedError

    def __call__(self, actions: torch.Tensor, raw_observations: torch.Tensor,
                 device: str, **kwargs):
        # Get rid of one-hot encoding of the task.
        observations = raw_observations[..., :self.OBS_DIM]
        rewards = self.get_reward(observations, actions, device, **kwargs)
        return rewards.view(-1, 1)
        # rewards = tf.TensorArray(tf.float32, size=observations.shape[0])
        # for i in tf.range(observations.shape[0]):
        #    rewards = rewards.write(
        #        i, self.get_reward(observations[i], actions[i]))
        # return rewards.stack()
