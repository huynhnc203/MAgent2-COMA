import numpy as np
import torch as th

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


class COMABuffer:
    def __init__(self, buffer_size, n_agents, obs_dim, action_dim, preprocessor=None):
        """
        Replay Buffer for COMA.
        :param buffer_size: Maximum number of experiences to store.
        :param n_agents: Number of agents.
        :param obs_dim: Dimension of observations.
        :param action_dim: Dimension of actions (for one-hot encoding).
        """
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.preprocessor = preprocessor

        # Initialize buffer components as numpy arrays
        self.observations = np.zeros((buffer_size, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_agents, action_dim), dtype=np.int32)
        self.next_observations = np.zeros((buffer_size, n_agents, obs_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.terminations = np.zeros((buffer_size, n_agents), dtype=np.float32)

        self.current_index = 0
        self.size = 0

    def add(self, obs, actions, next_obs, rewards, terminations):
        """
        Add a new experience to the buffer.
        :param obs: Observations [n_agents, obs_dim]
        :param actions: Actions [n_agents]
        :param next_obs: Next observations [n_agents, obs_dim]
        :param rewards: Rewards [n_agents]
        :param terminations: Termination flags [n_agents]
        """
        # Store data at the current index

        if self.preprocessor is not None:
            actions = self.preprocessor.transform(actions)

        self.observations[self.current_index] = obs
        self.actions[self.current_index] = actions
        self.next_observations[self.current_index] = next_obs
        self.rewards[self.current_index] = rewards
        self.terminations[self.current_index] = terminations

        # Update index and size
        self.current_index = (self.current_index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size, seq_len):
        """
        Sample a batch of sequential experiences from the buffer.
        :param batch_size: Number of sequences to sample.
        :param seq_len: Length of each sequence.
        :return: A batch of sequential experiences.
        """
        if self.size < seq_len:
            raise ValueError("Not enough data in buffer to sample the required sequence length.")

        # Sample starting indices for the sequences
        indices = np.random.choice(self.size - seq_len, batch_size, replace=False)

        # Collect sequences for each batch
        obs_batch = []
        actions_batch = []
        next_obs_batch = []
        rewards_batch = []
        terminations_batch = []

        for idx in indices:
            obs_batch.append(self.observations[idx: idx + seq_len])
            actions_batch.append(self.actions[idx: idx + seq_len])
            next_obs_batch.append(self.next_observations[idx: idx + seq_len])
            rewards_batch.append(self.rewards[idx: idx + seq_len])
            terminations_batch.append(self.terminations[idx: idx + seq_len])

        # Convert to PyTorch tensors
        obs_batch = th.tensor(np.array(obs_batch), dtype=th.float32)
        actions_batch = th.tensor(np.array(actions_batch), dtype=th.float32)
        next_obs_batch = th.tensor(np.array(next_obs_batch), dtype=th.float32)
        rewards_batch = th.tensor(np.array(rewards_batch), dtype=th.float32)
        terminations_batch = th.tensor(np.array(terminations_batch), dtype=th.float32)

        return obs_batch, actions_batch, next_obs_batch, rewards_batch, terminations_batch
    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return self.size



class COMAReplayBuffer:
    def __init__(self, buffer_size, seq_len, n_agents, obs_dim, action_dim, state_dim):
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_id = np.zeros((buffer_size, seq_len, n_agents, n_agents), dtype=np.int32)
        self.observations = np.zeros((buffer_size, seq_len, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((buffer_size, seq_len, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, seq_len, n_agents, action_dim), dtype=np.int32)
        self.rewards = np.zeros((buffer_size, seq_len, n_agents), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, seq_len, n_agents, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((buffer_size, seq_len, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, seq_len, n_agents), dtype=np.bool_)
        self.current_index = 0
        self.size = 0

    def add(self, agent_id, obs, state, actions, rewards, next_obs, next_state, dones):
        idx = self.current_index
        self.agent_id[idx] = agent_id
        self.observations[idx] = obs
        self.state[idx] = state
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_observations[idx] = next_obs
        self.next_state[idx] = next_state
        self.dones[idx] = dones


        self.current_index = (self.current_index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        agent_id = th.tensor(self.agent_id[indices], dtype=th.int64)
        obs_batch = th.tensor(self.observations[indices], dtype=th.float32)
        state_batch = th.tensor(self.state[indices], dtype=th.float32)
        actions_batch = th.tensor(self.actions[indices], dtype=th.int64)
        rewards_batch = th.tensor(self.rewards[indices], dtype=th.float32)
        next_obs_batch = th.tensor(self.next_observations[indices], dtype=th.float32)
        next_state_batch = th.tensor(self.next_state[indices], dtype=th.float32)
        dones_batch = th.tensor(self.dones[indices], dtype=th.float32)

        return agent_id, obs_batch, state_batch, actions_batch, rewards_batch, next_obs_batch, next_state_batch, dones_batch

    def __len__(self):
        return self.size