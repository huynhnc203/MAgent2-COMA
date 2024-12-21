from marl_src.components.replay_buffer import ReplayBuffer
from marl_src.components.schema import Step, State
import torch


class IA2CC:
    def __init__(self, critic, actors, critic_optimizer, actor_optimizers, cfg):
        self.critic = critic
        self.actors = actors
        self.critic_optimizer = critic_optimizer
        self.actor_optimizers = actor_optimizers
        self.cfg = cfg
        self.buffer = ReplayBuffer(capacity=self.cfg["buffer_capacity"])
        self.step = Step(self.cfg["num_entities"])

    def save(self, version: str = "0.0"):
        import os
        load_dir = os.path.join(f"ver-{version}")
        os.makedirs(load_dir, exist_ok=True)
        critic_path = os.path.join(load_dir, "critic.pt")
        torch.save(self.critic.state_dict(), critic_path)
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(load_dir, f"agent-{i}.pt")
            torch.save(actor.state_dict(), actor_path)

        print(f"Models loaded from directory: {load_dir}")

    def compute_loss(self, gamma, epsilon):
        if len(self.buffer) < self.cfg["batch_size"]:
            return 0.0, 0.0
        full_observations, full_actions, next_full_observations, full_rewards, rewards = self.buffer.sample(self.cfg["batch_size"])
        full_observations = torch.FloatTensor(full_observations).to(self.cfg["device"])
        full_actions = torch.LongTensor(full_actions).to(self.cfg["device"])
        next_full_observations = torch.FloatTensor(next_full_observations).to(self.cfg["device"])
        full_rewards = torch.FloatTensor(full_rewards).to(self.cfg["device"])
        rewards = torch.FloatTensor(rewards).to(self.cfg["device"])
        values = self.critic(full_observations)
        next_values = self.critic(next_full_observations).detach()
        td_target = rewards + gamma * next_values
        td_error = td_target - values

        critic_loss = td_error.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actors_loss = 0.0
        for i in range(cfg["num_entities"]):
            observations = full_observations[:, i]
            actions  = full_actions[:, i]
            next_observations = next_full_observations[:, i]
            rewards = full_rewards[:, i]
            # print(f"Observations: {observations.shape}")
            # print(f"Actions: {actions.shape}")
            # print(f"Next observations: {next_observations.shape}")
            # print(f"Rewards: {rewards.shape}")
            # observations = torch.FloatTensor(observations).to(self.cfg["device"])
            # actions = torch.LongTensor(actions).to(self.cfg["device"])
            # next_observations = torch.FloatTensor(next_observations).to(self.cfg["device"])
            # rewards = torch.FloatTensor(rewards).to(self.cfg["device"])

            action_probs = self.actors[i](observations)
            action_probs = action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

            log_prob = torch.log(action_probs + epsilon)
            actor_loss = -(log_prob * td_error.detach()).sum()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            actors_loss += actor_loss.item()

        return critic_loss.item(), actors_loss * 1. / self.cfg["num_agents"]



    def train(self, params, version: str = "0.2"):
        from magent2.environments import battle_v4
        env = battle_v4.env(
            map_size=45,
            minimap_mode=False,
            render_mode=None,
        )
        decay = params["theta_decay"]
        for episode in range(self.cfg["episodes"]):
            state = env.reset()
            done = False
            sum_rewards = 0.0
            exploration_rate = select_explore_rate(episode, 1.0, 0.01, decay)
            for agent in env.agent_iter():
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
                            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
                            with torch.no_grad():
                                q_values = self.actors[get_index(int(agent_handle[1]), cfg["num_agents"])](state_tensor)
                            action = torch.argmax(q_values, dim=1).item()
                            if action == 21:
                                  action = env.action_space(agent).sample()

                        env.step(action)
                        next_observation, reward, termination, truncation, _ = env.last()
                        sum_rewards += reward
                        self.step.set(int(agent_handle[1]), observation, action, next_observation, reward)
                        done = False
                    else:
                        if not done:
                            self.buffer.push(self.step.full_step())
                            self.step.reset()
                            done = True
                            critic_loss, actors_loss = self.compute_loss(params["gamma"], 1e-8)
                            self.record.add_actors_loss(actors_loss)
                            self.record.add_critic_loss(critic_loss)
                            # print(f"Critic Loss: {critic_loss}, actors: {actors_loss}")

                        action = env.action_space(agent).sample()
                        env.step(action)
            if not done:
                self.buffer.push(self.step.full_step())
                critic_loss, actors_loss = self.compute_loss(params["gamma"], 1e-8)
                # print(f"Critic Loss: {critic_loss}, actors: {actors_loss}")

            print(f"Episode: {episode}")

            if episode % 200 == 0:
                self.save(version)

            self.record.add_rewards(reward)