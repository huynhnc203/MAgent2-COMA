import torch
import copy
from marl_src.components.replay_buffer import ReplayBuffer
from marl_src.components.schema import (
    Step,
    State
)


class Agent:

    def get_action(self, agent_id):
        raise NotImplementedError("get_action method is not implemented")




class BaseSARL(Agent):

    def save_model(self, path):
        import os
        load_dir = os.path.join(path)
        print(load_dir)
        os.makedirs(load_dir, exist_ok=True)
        if self.agent is not None:
            torch.save(self.agent.state_dict(), "{}/agent.pt".format(path))

    def load_model(self, path):
        if self.agent is not None:
            self.agent.load_state_dict(
                torch.load(path, map_location=lambda storage, loc: storage))

    def compute_loss(self, gamma, epsilon):
        raise NotImplementedError("compute_loss method is not implemented")

    def train(self, params):
        raise NotImplementedError("train method is not implemented")

    def evaluate(self, params):
        raise NotImplementedError("evaluate method is not implemented")
