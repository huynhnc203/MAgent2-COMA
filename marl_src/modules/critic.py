import torch as th
import torch.nn as nn
import torch.nn.functional as F

class COMACritic(nn.Module):
    def __init__(self, args):

        super(COMACritic, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args["state_dim"] + args["n_actions"] * args["n_agents"], args["hidden_dim"])
        self.fc2 = nn.Linear(args["hidden_dim"], args["hidden_dim"])
        self.fc3 = nn.Linear(args["hidden_dim"], args["hidden_dim"])
        self.value_head = nn.Linear(args["hidden_dim"], args["n_agents"] * args["n_actions"])

    def forward(self, state, actions):
        x = F.relu(self.fc1(self._build_input(state, actions)))      # First layer with ReLU
        x = F.relu(self.fc2(x))           # Second layer with ReLU
        x = F.relu(self.fc3(x))           # Third layer with Re
        v = self.value_head(x)            # Value function prediction
        return v

    def _build_input(self, states, actions):
        return th.cat([states, actions.view(-1, self.args["n_agents"] * self.args["n_actions"])], dim=1)