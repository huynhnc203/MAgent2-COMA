import json
import os

class RLTrainingRecord:
    def __init__(self, save_path="metrics.json"):
        """
        Initialize the RLTrainingRecord class.

        Args:
            save_path (str): Path to save the metrics in JSON format.
        """
        self.save_path = f"metrics/{save_path}"
        self.metrics = {
            "episodes": [],
            "rewards": [],
            "actor_losses": [],
            "critic_losses": [],
            "episode_lengths": [],
        }

    def record_episode(self, episode, reward, episode_length):
        """
        Record metrics for an episode.

        Args:
            episode (int): Current episode number.
            reward (float): Total reward obtained in the episode.
            episode_length (int): Length of the episode in steps.
        """
        self.metrics["episodes"].append(episode)
        self.metrics["rewards"].append(reward)
        self.metrics["episode_lengths"].append(episode_length)

    def record_loss(self, actor_loss=None, critic_loss=None):
        """
        Record loss values for actor and critic during training.

        Args:
            actor_loss (float, optional): Loss for the actor network.
            critic_loss (float, optional): Loss for the critic network.
        """
        if actor_loss is not None:
            self.metrics["actor_losses"].append(actor_loss)
        if critic_loss is not None:
            self.metrics["critic_losses"].append(critic_loss)

    def save_metrics(self):
        """
        Save the metrics to a JSON file.
        """
        with open(self.save_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {self.save_path}")

    def load_metrics(self):
        """
        Load metrics from a JSON file.

        Returns:
            dict: The loaded metrics.
        """
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                self.metrics = json.load(f)
            print(f"Metrics loaded from {self.save_path}")
        else:
            print(f"No existing metrics found at {self.save_path}. Starting fresh.")
        return self.metrics

    def print_latest_metrics(self):
        """
        Print the most recent metrics for quick reference.
        """
        if self.metrics["episodes"]:
            print(f"Episode: {self.metrics['episodes'][-1]}")
            print(f"Reward: {self.metrics['rewards'][-1]:.2f}")
            print(f"Episode Length: {self.metrics['episode_lengths'][-1]} steps")
        if self.metrics["actor_losses"]:
            print(f"Actor Loss: {self.metrics['actor_losses'][-1]:.4f}")
        if self.metrics["critic_losses"]:
            print(f"Critic Loss: {self.metrics['critic_losses'][-1]:.4f}")