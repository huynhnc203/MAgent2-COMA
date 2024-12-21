import numpy as np

class State:
    def __init__(self, observation, action, next_observation, reward):
        self.observation = observation
        self.action = action
        self.next_observation = next_observation
        self.reward = reward

    def set(self, observation, action, next_observation, reward):
        self.observation = observation
        self.action = action
        self.next_observation = next_observation
        self.reward = reward

    def tuple_return(self):
        return (self.observation, self.action, self.next_observation, self.reward)

class Step:
    def __init__(self, num_agents):
        self.agent_states = [
            State(
                observation = np.zeros((13, 13, 5)),
                action = 21.0,
                next_observation = np.zeros((13, 13, 5)),
                reward = 0.0
            )
            for _ in range(num_agents)
        ]
        self.num_agents = num_agents
        self.rewards = 0.0

    def set(self, index, observation, action, next_observation, reward):
        self.agent_states[index].set(observation, action, next_observation, reward)
        self.rewards += reward

    def reset(self):
        self.agent_states = [
            State(
                observation = np.zeros((13, 13, 5)),
                action = 21.0,
                next_observation = np.zeros((13, 13, 5)),
                reward = 0.0
            )
            for _ in range(self.num_agents)
        ]
        self.rewards = 0.0

    def full_step(self):
        full_observation = np.array([state.observation for state in self.agent_states])
        next_full_observation = np.array([state.next_observation for state in self.agent_states])
        full_actions = np.array([state.action for state in self.agent_states])
        full_rewards = np.array([state.reward for state in self.agent_states])
        return (full_observation, full_actions, next_full_observation, full_rewards, self.rewards)