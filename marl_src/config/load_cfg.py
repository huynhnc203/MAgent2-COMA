import yaml

def load_cfg(cfg_path):
    import os

    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, cfg_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    args = config.get("args", {})
    args["obs_dim"] = 13 * 13 * 5
    env_cfg = config.get("env", {})
    training_cfg = config.get("training_cfg", {})
    model = config.get("model")
    return args, env_cfg, training_cfg, model



def get_abs_path(path):
    import os
    return os.path.join(os.path.dirname(__file__), path)