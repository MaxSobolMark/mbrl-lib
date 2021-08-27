import torch
from mbrl.util.lifelong_learning import general_reward_function

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_general_reward_function():
    def reward_function_1(act: torch.Tensor, obs: torch.Tensor):
        assert len(act.shape) == len(obs.shape) == 2
        return torch.ones(obs.shape[:-1]).view(-1, 1)

    def reward_function_2(act: torch.Tensor, obs: torch.Tensor):
        assert len(act.shape) == len(obs.shape) == 2
        return torch.ones(obs.shape[:-1]).view(-1, 1) * 2

    def reward_function_3(act: torch.Tensor, obs: torch.Tensor):
        assert len(act.shape) == len(obs.shape) == 2
        return torch.ones(obs.shape[:-1]).view(-1, 1) * 3

    observations = torch.Tensor([
        [0, 1, 2, 3, 4, 1, 0, 0],
        [4, 3, 2, 1, 0, 0, 0, 1],
        [10, 11, 12, 13, 14, 1, 0, 0],
        [12, 11, 10, 13, 9, 0, 1, 0],
        [5, 4, 3, 5, 1, 0, 0, 1],
    ]).to(_DEVICE)
    act = torch.rand([observations.shape[0], 6]).to(_DEVICE)
    expected_rewards = torch.Tensor([1, 3, 1, 2, 3]).view(-1, 1)
    rewards = general_reward_function(
        act, observations,
        [reward_function_1, reward_function_2, reward_function_3])
    assert torch.equal(expected_rewards, rewards)
