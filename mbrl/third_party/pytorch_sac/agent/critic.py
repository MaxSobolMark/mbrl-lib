import torch
from torch import nn

from mbrl.third_party.pytorch_sac import utils
from mbrl.third_party.pnn.pnn import PNN

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f"train_critic/{k}_hist", v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f"train_critic/q1_fc{i}", m1, step)
                logger.log_param(f"train_critic/q2_fc{i}", m2, step)


class PNNDoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, device):
        super().__init__()
        # self.sizes = [obs_dim + action_dim, hidden_dim, hidden_dim, 1]
        self.Q1 = PNN(obs_dim + action_dim, hidden_dim, 128, 1, device)
        self.Q2 = PNN(obs_dim + action_dim, hidden_dim, 128, 1, device)
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1.forward(obs_action)
        q2 = self.Q2.forward(obs_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def new_task(self):
        self.Q1.new_task()
        self.Q2.new_task()
        print('critic new task added')

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f"train_critic/{k}_hist", v, step)

        # assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1.columns, self.Q2.columns)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f"train_critic/q1_fc{i}", m1, step)
                logger.log_param(f"train_critic/q2_fc{i}", m2, step)