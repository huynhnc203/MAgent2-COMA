
from marl_src.components.replay_buffer import COMAReplayBuffer
from marl_src.controller.mac import MAC
from marl_src.modules.actor import RNNAgent
from marl_src.modules.critic import COMACritic
from marl_src.learner.coma import COMALearner
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_model import QNetwork

def train(args, env_cfg, training_cfg, model, test_mode=False):
    writer = SummaryWriter()
    buffer = COMAReplayBuffer(
        args["buffer_size"], env_cfg["max_cycles"], args["n_agents"], args["obs_dim"], args["n_actions"], args["state_dim"]
    )
    mac = MAC(input_shape=args["obs_dim"] + args["n_agents"], args=args)

    q_net = QNetwork((13, 13, 5), 21)
    q_net.load_state_dict(torch.load("red.pt", weights_only=True, map_location="cpu"))

    # Set up COMA Critic
    critic = COMACritic(args)
    learner = COMALearner(mac, critic, buffer, args)

    from magent2.environments import battle_v4
    env = battle_v4.env(map_size=45,
                        minimap_mode=False,
                        render_mode=None,
                        max_cycles=env_cfg["max_cycles"],
                        step_reward=env_cfg["step_reward"],
                        dead_penalty=env_cfg["dead_penalty"],
                        attack_penalty=env_cfg["attack_penalty"],
                        attack_opponent_reward=env_cfg["attack_opponent_reward"],
                        extra_features=False)

    device = training_cfg["device"]

    for episode in range(training_cfg["episodes"]):
        episode_start_time = time.time()
        state = env.reset()

        n_agents = args["n_agents"]
        max_cycles = env_cfg["max_cycles"]

        # Buffers to store episode data
        agent_ids = np.zeros((max_cycles, n_agents, args["n_agents"]))
        observations = np.zeros((max_cycles, n_agents, args["obs_dim"]))
        state = np.zeros((max_cycles, args["state_dim"]))
        actions = np.zeros((max_cycles, n_agents, args["n_actions"]), dtype=np.int32)
        rewards = np.zeros((max_cycles, n_agents))
        next_state = np.zeros((max_cycles, args["state_dim"]))
        next_observations = np.zeros((max_cycles, n_agents, args["obs_dim"]))
        dones = np.zeros((max_cycles, n_agents))

        timestep = 0
        mac.init_hidden_one()
        blue = False
        ep_actions = []
        reward_sum = 0
        exploration_rate = select_explore_rate(episode, 0.7, 0.3, 0.0001)
        red_lives = {f"red_{index}": True for index in range(81)}
        blue_lives = {f"blue_{index}": True for index in range(81)}
        for i, agent in enumerate(env.agent_iter()):
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
            else:
                agent_handle = agent.split("_")

                if agent_handle[0] == "blue":
                    agent_id = int(agent_handle[1])
                    agent_id_one_hot = torch.nn.functional.one_hot(torch.tensor(agent_id), num_classes=args["n_agents"])
                    agent_ids[timestep, agent_id] = agent_id_one_hot.numpy()
                    observations[timestep, agent_id] = observation.reshape(-1)
                    state[timestep] = env.state().reshape(-1)
                    obs_tensor = torch.FloatTensor(observation.reshape(-1)).unsqueeze(0).to(device)

                    if np.random.random() < exploration_rate:
                        action = torch.tensor(np.random.randint(0, args["n_actions"] - 1))
                    else:
                        action = mac.select_actions(obs_tensor, agent_id_one_hot, test_mode=True)
                    ep_actions.append(action.item())
                    actions[timestep, agent_id] = torch.nn.functional.one_hot(action, num_classes=args["n_actions"]).numpy()
                    env.step(action.item())
                    new_observation, reward, termination, truncation, _ = env.last()
                    next_observations[timestep, agent_id] = new_observation.reshape(-1)
                    try:
                        next_state[timestep] = env.state().reshape(-1)
                    except:
                        is_blue_lives, is_red_lives, after_game_reward = 0, 0, args["end_game_reward"]
                        for key, value in blue_lives.items():
                            is_blue_lives += value
                        for key, value in red_lives.items():
                            is_red_lives += value
                        print("Done game")
                        rewards[timestep, agent_id] = reward + after_game_reward
                    rewards[timestep, agent_id] = reward
                    dones[timestep, agent_id] = 1 - termination
                    reward_sum += reward
                    blue = True
                else:
                    if blue:
                        timestep += 1
                        blue = False
                    observation = (
                        torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                    )
                    if np.random.random() < 0.2:
                        with torch.no_grad():
                            action = q_net(observation).argmax().item()
                    else:
                        action = env.action_space(agent).sample()
                    env.step(action)


        writer.add_scalar("Reward", reward_sum, episode)
        # Add the episode to the buffer
        buffer.add(agent_ids, observations, state, actions, rewards, next_observations, next_state, dones)

        # Train COMA Learner
        learner.train(args["batch_size"], writer, episode)

        episode_end_time = time.time()
        print(f"Episode: {episode} | Duration: {episode_end_time - episode_start_time:.2f}s")
        if (episode + 1) % args["save_frequency"] == 0:
            learner.save_models(f"coma-models")
            win = 0
            new_buffer = COMAReplayBuffer(
                args["buffer_size"], env_cfg["max_cycles"], args["n_agents"], args["obs_dim"], args["n_actions"],
                args["state_dim"]
            )
            new_mac = MAC(input_shape=args["obs_dim"] + args["n_agents"], args=args)
            new_critic = COMACritic(args)
            new_learner = COMALearner(new_mac, new_critic, new_buffer, args)
            new_learner.load_models(f"coma-models")
            for games in range(5):
                env.reset()
                new_mac.init_hidden_one()
                red_lives = {f"red_{index}": True for index in range(81)}
                blue_lives = {f"blue_{index}": True for index in range(81)}
                for i, agent in enumerate(env.agent_iter()):
                    observation, reward, termination, truncation, info = env.last()
                    if termination or truncation:
                        if termination:
                            if agent.split("_")[0] == "blue":
                                blue_lives[agent] = False
                            else:
                                red_lives[agent] = False
                        env.step(None)
                    else:
                        agent_handle = agent.split("_")
                        if agent_handle[0] == "blue":
                            agent_id = int(agent_handle[1])
                            agent_id_one_hot = torch.nn.functional.one_hot(torch.tensor(agent_id), num_classes=args["n_agents"])
                            obs_tensor = torch.FloatTensor(observation.reshape(-1)).unsqueeze(0).to(device)
                            action = new_mac.select_actions(obs_tensor, agent_id_one_hot, test_mode=True)
                            env.step(action.item())
                        else:
                            env.step(env.action_space(agent).sample())
                is_blue_lives, is_red_lives = 0, 0
                for key, value in blue_lives.items():
                    is_blue_lives += value
                for key, value in red_lives.items():
                    is_red_lives += value
                if is_red_lives == 0:
                    win += 1
                print(f"Blue: {is_blue_lives}, Red: {is_red_lives}")
            print(f"Win rate: {win / 5}")


def select_explore_rate(episode, initial_rate, final_rate, decay):
    return max(final_rate, initial_rate - episode * decay)