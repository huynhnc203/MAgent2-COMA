import random

from .base import BaseSARL
import numpy as np
from marl_src.components.replay_buffer import ReplayBuffer
from marl_src.components.record import RLTrainingRecord
from marl_src.modules.agent import QNetwork

import copy
import torch.optim as opt
import torch

from marl_src.utils import select_explore_rate

import time

class QLearner(BaseSARL):
    def __init__(self, cfg = None, env_cfg = None, version: str = "0.0"):
        super().__init__()
        self.agent = QNetwork((13, 13, 5), cfg["num_actions"]).to(cfg["device"])
        self.agent_target = copy.deepcopy(self.agent)
        self.agent_optimizer = opt.Adam(self.agent.parameters(), lr=cfg["lr"])
        self.cfg = cfg
        self.buffer = ReplayBuffer(capacity=self.cfg["buffer_capacity"])
        self.version = version
        self.env_cfg = env_cfg

    def compute_loss(self, gamma, epsilon, tau, target_update_frequency, global_step):
        if len(self.buffer) < self.cfg["batch_size"]:
            return 10000

        observations, actions, next_observations, rewards, dones = self.buffer.sample(self.cfg["batch_size"])
        device = self.cfg["device"]

        observations = torch.FloatTensor(observations).permute([0, 3, 1, 2]).to(device)
        actions = torch.LongTensor(actions).to(self.cfg["device"])
        next_observations = torch.FloatTensor(next_observations).permute([0, 3, 1, 2]).to(device)
        rewards = torch.FloatTensor(rewards).to(self.cfg["device"])
        dones = torch.LongTensor(dones).to(self.cfg["device"])


        with torch.no_grad():
            next_values, indices = self.agent_target(next_observations).max(dim=1)
            td_target = rewards.flatten() + gamma * next_values * (1 - dones.flatten())
        actions = actions.unsqueeze(-1)
        old_value = self.agent(observations).gather(1, actions).squeeze()

        loss = torch.nn.functional.mse_loss(old_value, td_target)
        self.agent_optimizer.zero_grad()
        loss.backward()
        self.agent_optimizer.step()

        if target_update_frequency is not None and global_step is not None:

            if global_step % target_update_frequency == 0:
                self.agent_target.load_state_dict(self.agent.state_dict())

        if tau is not None:
            for target_param, param in zip(self.agent_target.parameters(), self.agent.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        return loss.item()

        # critic_loss = td_error.pow(2).mean()
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

    def train(self, params):
        from magent2.environments import battle_v4
        env = battle_v4.env(map_size=45,
                            minimap_mode=False,
                            render_mode=None,
                            max_cycles=self.env_cfg["max_cycles"],
                            step_reward=self.env_cfg["step_reward"],
                            dead_penalty=self.env_cfg["dead_penalty"],
                            attack_penalty=self.env_cfg["attack_penalty"],
                            attack_opponent_reward=self.env_cfg["attack_opponent_reward"],
                            extra_features=False)
        count, sum_loss, training_count = 0, 0, 0
        device = self.cfg["device"]
        for episode in range(self.cfg["episodes"]):
            episode_start_time = time.time()
            state = env.reset()
            # print(state)
            done = False
            sum_rewards = 0.0
            exploration_rate = select_explore_rate(self.cfg["episodes"], episode, params["min_exploration"],
                                                   params["max_exploration"])

            for i, agent in enumerate(env.agent_iter()):
                observation, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    action = None
                    env.step(action)
                else:
                    agent_handle = agent.split("_")
                    if agent_handle[0] == "blue":
                        if np.random.random() < exploration_rate:
                            action = env.action_space(agent).sample()
                        else:
                            state_tensor = torch.FloatTensor(observation).permute([2, 0, 1]).unsqueeze(0).to(device)
                            with torch.no_grad():
                                q_values = self.agent(state_tensor)
                            action = torch.argmax(q_values, dim=1).item()

                        env.step(action)
                        next_observation, reward, termination, truncation, _ = env.last()
                        sum_rewards += reward
                        self.buffer.push((observation, action, next_observation, reward, termination))
                        count += 1
                        if count % params["training_frequency"] == 0:
                            loss = self.compute_loss(params["gamma"], params["epsilon"], params["tau"],
                                                     params["target_update_frequency"], count)
                            sum_loss += loss
                            training_count += 1
                    else:
                        action = env.action_space(agent).sample()
                        env.step(action)
            # print(env.agents)
            if episode % params["save_frequency"] == 0:
                self.save_model(params["save_path"])
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            print("LOG Episode: {} | Loss: {:.4f} | Reward: {:.4f} | Time: {:.2f} seconds".format(
                episode, sum_loss / training_count, sum_rewards, episode_duration
            ))

    def get_action(self, observation):
        state_tensor = torch.FloatTensor(observation).permute([2, 0, 1]).unsqueeze(0).to(self.cfg["device"])
        with torch.no_grad():
            q_values = self.agent(state_tensor)
        return torch.argmax(q_values, dim=1).item()



