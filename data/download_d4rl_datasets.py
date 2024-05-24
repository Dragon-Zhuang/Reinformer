import os
import gym
import numpy as np

import collections
import pickle

import d4rl

def download_d4rl_data():

    data_dir = "d4rl_dataset/"

    print(data_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Gym-v2
    for env_name in ["walker2d","halfcheetah","hopper"]:
        for dataset_type in ["medium-v2", "medium-replay-v2", "medium-expert-v2"]: 
    # Maze2d-v1
    # for env_name in ["maze2d"]:
        # for dataset_type in ["umaze-v1", "medium-v1", "large-v1"]:
    # Antmaze-v2
    # for env_name in ["antmaze"]:
        # for dataset_type in ["umaze-v2", "umaze-diverse-v2", "medium-play-v2", "medium-diverse-v2", "large-play-v2", "large-diverse-v2"]:
    # Ktichen-v0
    # for env_name in ["kitchen"]:
    #     for dataset_type in ["complete-v0", "partial-v0", "mixed-v0"]:

            name = f"{env_name}-{dataset_type}"
            pkl_file_path = os.path.join(data_dir, name)

            print("processing: ", name)

            env = gym.make(name)
            dataset = d4rl.qlearning_dataset(env)

            N = dataset["rewards"].shape[0]
            data_ = collections.defaultdict(list)

            use_timeouts = False
            if "timeouts" in dataset:
                use_timeouts = True

            episode_step = 0
            paths = []
            for i in range(N):
                done_bool = bool(dataset["terminals"][i])
                if use_timeouts:
                    final_timestep = dataset["timeouts"][i]
                else:
                    final_timestep = (episode_step == 1000-1)
                for k in ["observations", "next_observations", "actions", "rewards", "terminals"]:
                    data_[k].append(dataset[k][i])
                if done_bool or final_timestep:
                    episode_step = 0
                    episode_data = {}
                    for k in data_:
                        episode_data[k] = np.array(data_[k])
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1

            returns = np.array([np.sum(p["rewards"]) for p in paths])
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            print(f"Number of samples collected: {num_samples}")
            print(f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}")

            with open(f"{pkl_file_path}.pkl", "wb") as f:
                pickle.dump(paths, f)


if __name__ == "__main__":
    download_d4rl_data()
