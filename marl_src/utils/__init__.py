

def select_explore_rate(max_episode, episode, min_rate, max_rate):
    return max_rate - (max_rate - min_rate) * episode / max_episode