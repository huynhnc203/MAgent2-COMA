from magent2.environments import battle_v4
import os
import cv2
from marl_src.learner.coma import COMALearner
from marl_src.modules.critic import COMACritic
from marl_src.controller.mac import MAC
from marl_src.components.replay_buffer import COMAReplayBuffer
from marl_src.config.load_cfg import load_cfg
import torch

def eval(env, new_mac, pretrain, type="pretrain"):
    win_count, draw_count, lose_count = 0, 0, 0
    for game in range(30):
        env.reset()
        red_lives = {f"red_{index}": True for index in range(81)}
        blue_lives = {f"blue_{index}": True for index in range(81)}
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                if termination:
                    # print(agent.split("_")[0])
                    if agent.split("_")[0] == "blue":
                        blue_lives[agent] = False
                    else:
                        red_lives[agent] = False
                action = None
            else:
                agent_handle = agent.split("_")[0]
                agent_id = int(agent.split("_")[1])
                if agent_handle == "red":
                    with torch.no_grad():
                        agent_id_one_hot = torch.nn.functional.one_hot(torch.tensor(agent_id),
                                                                       num_classes=args["n_agents"])
                        obs_tensor = torch.FloatTensor(observation.reshape(-1)).unsqueeze(0)
                        action = new_mac.select_actions(obs_tensor, agent_id_one_hot, test_mode=True).item()
                    # print(action)
                    # print(action.item())
                    try:
                        env.state()
                    except:
                        win_count += 1
                else:
                    if type == "pretrain":
                        observation = (
                            torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                        )
                        with torch.no_grad():
                            action = pre_train(observation).argmax().item()
                    else:
                        action = env.action_space(agent).sample()

                    try:
                        env.state()
                    except:
                        lose_count += 1
            env.step(action)
        is_blue_lives, is_red_lives = 0, 0
        for key, value in blue_lives.items():
            is_red_lives += value
        for key, value in red_lives.items():
            is_blue_lives += value
        print(is_blue_lives, is_red_lives)
        if is_red_lives > 0 and is_blue_lives == 0:
            draw_count += 1
    print(f"{type}: Win: {win_count}, Lose: {lose_count}, Draw: {draw_count}")

if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    env.reset()
    args, env_cfg, training_cfg, model = load_cfg("coma.yaml")
    new_buffer = COMAReplayBuffer(
        args["buffer_size"], env_cfg["max_cycles"], args["n_agents"], args["obs_dim"], args["n_actions"],
        args["state_dim"]
    )
    new_mac = MAC(input_shape=args["obs_dim"] + args["n_agents"], args=args)
    new_critic = COMACritic(args)
    new_learner = COMALearner(new_mac, new_critic, new_buffer, args)
    print(os.path.join(os.path.dirname(__file__), f"models/coma-models-{9}"))
    indice = [5, 7, 10, 11, 12]
    for index in indice:
        new_learner.load_models(f"coma-models-{index}")
        new_mac.init_hidden_one()
        from final_torch_model import QNetwork
        from torch_model import QNetwork as Nq

        pre_train_final = QNetwork(
            env.observation_space("red_0").shape, env.action_space("red_0").n
        )
        pre_train_final.load_state_dict(torch.load("red_final.pt", weights_only=True, map_location="cpu"))

        pre_train = Nq(
            env.observation_space("blue_0").shape, env.action_space("blue_0").n
        )
        pre_train.load_state_dict(torch.load("red.pt", weights_only=True, map_location="cpu"))

        eval(env, new_mac, pre_train, "random")
        eval(env, new_mac, pre_train, "pretrain")
        eval(env, new_mac, pre_train_final, "pretrain")
    env.close()
