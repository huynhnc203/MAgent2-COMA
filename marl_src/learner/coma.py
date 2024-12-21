import torch as th
import torch.nn.functional as F
import copy
import os

class COMALearner:
    def __init__(self, mac, critic, buffer, args):
        self.mac = mac
        self.critic = critic
        self.critic_target = copy.deepcopy(critic)
        self.buffer = buffer
        self.args = args

        self.actor_optimizer = th.optim.Adam(self.mac.agent.parameters(), lr=args["lr"])
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=args["lr"])

    def train(self, batch_size, writer, episode):
        if len(self.buffer) < batch_size:
            return

        # Sample batch data
        agent_ids, obs, state, actions, rewards, next_obs, next_state, dones = self.buffer.sample(batch_size)
        seq_len = obs.size(1)

        # Initialize hidden states
        self.mac.init_hidden(batch_size)
        total_actor_loss, total_critic_loss = 0, 0

        for t in range(seq_len):
            # Forward pass through the MAC
            mac_out = self.mac.forward(obs[:, t], agent_ids[:, t])  # Shape: [batch_size, n_agents, n_actions]
            mac_out = mac_out.view(batch_size, self.args["n_agents"], -1)
            q_val = self.critic(state[:, t], actions[:, t])  # Shape: [batch_size, n_agents * n_actions]
            q_val_chosen = q_val * actions[:, t].view(batch_size, -1)  # Shape: [batch_size, n_agents * n_actions]
            q_val_chosen = q_val_chosen.view(batch_size, self.args["n_agents"], -1).sum(dim=2)  # Shape: [batch_size, n_agents]

            with th.no_grad():
                # Calculate target Q value
                next_actions = self.mac.forward(next_obs[:, t], agent_ids[:, t])  # Next actions from target policy
                next_actions = next_actions.view(batch_size, self.args["n_agents"], -1)
                next_pi = F.softmax(next_actions, dim=2)  # (batch_size, n_agents, n_actions)
                next_actions_one_hot = th.argmax(next_pi, dim=2, keepdim=True)
                expanded_dones = dones[:, t].unsqueeze(-1)
                expanded_dones = expanded_dones.expand(-1, -1, self.args["n_actions"])  # Shape becomes (batch, n_agents, action_dims)
                # print(F.one_hot(next_actions_one_hot, self.args["n_actions"]).float().squeeze().shape, expanded_dones.shape)
                next_actions_one_hot = F.one_hot(next_actions_one_hot, self.args["n_actions"]).float().squeeze() * expanded_dones
                v = self.critic_target(next_state[:, t], next_actions_one_hot) * next_actions_one_hot.view(self.args["batch_size"], -1)  # Shape: [batch_size, n_agents * n_actions]
                v = v.view(self.args["batch_size"], self.args["n_agents"], -1).sum(dim=2)
                # print(rewards[: t].sum(dim=1).unsqueeze(-1))
                q_target = rewards[:, t].sum(dim=1).unsqueeze(-1).expand(-1, self.args["n_agents"]) + self.args["gamma"] * v # Shape: [batch_size, n_agents]

            # Critic loss
            critic_loss = F.mse_loss(q_val_chosen.view(batch_size, -1), q_target.view(batch_size, -1))

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute Advantage
            pi = F.softmax(mac_out, dim=2)  # (batch_size, n_agents, n_actions)
            q_val = q_val.detach().view(batch_size, self.args["n_agents"], -1) # (batch_size, n_agents, n_actions)
            baseline = (pi * q_val).sum(dim=2)  # (batch_size, n_agents)
            pi_taken =  th.clamp((pi * actions[:, t]).sum(dim=2), 1e-8) # (batch_size, n_agents)
            # print((baseline - q_val_chosen.detach()).mean())
            # print(th.log(pi_taken).mean())
            actor_loss = ((- th.log(pi_taken) * (baseline - q_val_chosen.detach())).sum(dim=1) / dones[:, max(0, t - 1)].sum()).mean()  # (batch_size, n_agents, n_actions)


            # actor_loss = q_taken.sum(dim=1).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # advantage = q_val.detach().squeeze(-1) - q_target  # Detach Q-value to avoid updating it during actor optimization
            #
            # log_pi = (pi * actions[:, t]).sum(dim=2).clamp(1e-8).log() #(batch_size, n_agents, 21)
            #
            # actor_loss = ((log_pi * advantage.unsqueeze(-1)) * dones[:, t]).sum() / dones[:, t].sum()
            #
            # total_actor_loss += actor_loss
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

            if t % self.args["target_update_frequency"] == 0:
                self._soft_update_target(self.args["tau"])
        writer.add_scalar("Loss/actor_Loss", total_actor_loss, episode)
        writer.add_scalar("Loss/critic_Loss", total_critic_loss, episode)
        print(f"Actor Loss: {total_actor_loss:.4f}, Critic Loss: {total_critic_loss:.4f}")

    def _soft_update_target(self, tau=0.005):
        """
        Softly update the target critic network.
        """
        for target_param, critic_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(critic_param.data)

    def save_models(self, save_dir):
        """
        Save the actor (MAC), critic, and their optimizers.
        """
        os.makedirs(save_dir, exist_ok=True)

        th.save(self.mac.agent.state_dict(), os.path.join(save_dir, "actor.pth"))
        th.save(self.actor_optimizer.state_dict(), os.path.join(save_dir, "actor_optimizer.pth"))

        th.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
        th.save(self.critic_optimizer.state_dict(), os.path.join(save_dir, "critic_optimizer.pth"))

        print(f"Models saved successfully to {save_dir}")

    def load_models(self, load_dir):
        """
        Load the actor (MAC), critic, and their optimizers.
        """
        self.mac.agent.load_state_dict(th.load(os.path.join(load_dir, "actor.pth")))
        self.actor_optimizer.load_state_dict(th.load(os.path.join(load_dir, "actor_optimizer.pth")))

        # self.critic.load_state_dict(th.load(os.path.join(load_dir, "critic.pth")))
        # self.critic_optimizer.load_state_dict(th.load(os.path.join(load_dir, "critic_optimizer.pth")))
        #
        # self.critic_target = copy.deepcopy(self.critic)

        print(f"Models loaded successfully from {load_dir}")