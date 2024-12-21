from marl_src.modules.actor import RNNAgent
import torch.nn.functional as F
import torch as th

class MAC:
    def __init__(self, input_shape, args):
        self.args = args
        self.n_agents = args["n_agents"]
        self.agent = RNNAgent(input_shape, args)
        self.hidden_states = None

    def select_actions(self, obs, agent_id, test_mode=False):
        agent_id = agent_id.unsqueeze(0)
        obs = th.cat([obs, agent_id], dim=-1)
        with th.no_grad():
            agent_outs, self.hidden_states = self.agent(obs, self.hidden_states)
        if test_mode:
            return agent_outs.argmax(dim=-1)
        action_probs = F.softmax(agent_outs, dim=-1)
        return th.distributions.Categorical(action_probs).sample()

    def print_distribution(self, obs, agent_id, step, writer=None):
        agent_id_one_hot = th.nn.functional.one_hot(th.tensor(agent_id), self.n_agents).float().unsqueeze(0)
        obs = th.cat([obs, agent_id_one_hot], dim=-1)
        agent_outs, self.hidden_states = self.agent(obs, self.hidden_states)
        action_probs = F.softmax(agent_outs, dim=-1)
        with open("action_probs.txt", "a") as f:
            f.write(f"{action_probs}\n")
        # writer.add_tensor(f"Action Distribution/{agent_id}", action_probs, step)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().expand(batch_size, self.n_agents, -1)
        print(self.hidden_states.shape)

    def init_hidden_one(self):
        self.hidden_states = self.agent.init_hidden()

    def forward(self, obs, agent_id):
        obs = th.cat([obs, agent_id], dim=-1)
        agent_outs, self.hidden_states = self.agent(obs, self.hidden_states)
        return agent_outs