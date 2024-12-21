import argparse
from marl_src import train
from marl_src.config.load_cfg import load_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning Training Script")
    parser.add_argument("--config", type=str, default="q_learning.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--save", type=str, default="coma_0", help="Save Folder")
    args = parser.parse_args()

    args, env_cfg, training_cfg, model = load_cfg(args.config)

    print(f"Global Config: {args}")
    print(f"Environment Config: {env_cfg}")
    print(f"Training Config: {training_cfg}")
    print(f"Model: {model}")

    train(args, env_cfg, training_cfg, model, test_mode=False)
