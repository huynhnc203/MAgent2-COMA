import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args["rnn_hidden_dim"])
        self.layer_norm1 = nn.LayerNorm(args["rnn_hidden_dim"])

        self.rnn = nn.GRUCell(args["rnn_hidden_dim"], args["rnn_hidden_dim"])
        self.dropout = nn.Dropout(p=args.get("dropout", 0.2))

        self.fc2 = nn.Linear(args["rnn_hidden_dim"], args["n_actions"])
        self.layer_norm2 = nn.LayerNorm(args["n_actions"])

    def init_hidden(self, batch_size=None):
        if batch_size is not None:
            return self.fc1.weight.new(1, self.args["rnn_hidden_dim"]).zero_()
        return self.fc1.weight.new(1, self.args["rnn_hidden_dim"]).zero_()

    def forward(self, inputs, hidden_state):
        if len(inputs.shape) > 2:
            inputs = inputs.reshape(
                self.args["batch_size"] * self.args["n_agents"],
                self.args["obs_dim"] + self.args["n_agents"]
            )

        x = F.relu(self.layer_norm1(self.fc1(inputs))).squeeze(0)

        h_in = hidden_state.reshape(-1, self.args["rnn_hidden_dim"]).squeeze(0)
        h = self.rnn(x, h_in)

        h = self.dropout(h)
        q = self.layer_norm2(self.fc2(h))

        return q, h