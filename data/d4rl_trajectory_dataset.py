import pickle
import random
import numpy as np
import torch

from torch.utils.data import Dataset



def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


class D4RLTrajectoryDataset(Dataset):
    def __init__(
        self, 
        env_name,
        dataset_path, 
        context_len, 
        device
    ):

        self.context_len = context_len
        self.device = device
        # load dataset
        with open(dataset_path, "rb") as f:
            self.trajectories = pickle.load(f)

        # reward scale
        if env_name in ["hopper", "walker2d"]:
            scale = 1000
        elif env_name in ["halfcheetah"]:
            scale = 5000
        elif env_name in ["maze2d", "kitchen"]:
            scale = 100
        elif env_name in ["pen", "door", "hammer", "relocate"]:
            scale = 10000
        elif env_name in ["antmaze"]:
            scale = 1

        # calculate state mean and variance and returns_to_go for all traj
        states, returns, returns_to_go = [], [], []
        for traj in self.trajectories:
            if "antmaze" in dataset_path:
                # reward modification for antmaze
                traj["rewards"] = traj["rewards"] * 100 + 1
            states.append(traj["observations"])
            returns.append(traj["rewards"].sum())
            # calculate returns to go 
            traj["returns_to_go"] = (
                discount_cumsum(traj["rewards"], 1) / scale
            )
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = (
            np.mean(states, axis=0),
            np.std(states, axis=0) + 1e-6,
        )

        # normalize states
        for traj in self.trajectories:
            traj["observations"] = (
                traj["observations"] - self.state_mean
            ) / self.state_std

        # calculate returns max, mean, std
        returns = np.array(returns)
        self.return_stats = [
            returns.max(),
            returns.mean(),
            returns.std()
        ]
        print(f"dataset size: {len(self.trajectories)}\nreturns max : {returns.max()}\nreturns mean: {returns.mean()}\nreturns std : {returns.std()}")

    def get_state_stats(self):
        return self.state_mean, self.state_std
    
    def get_return_stats(self):
        return self.return_stats

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj["observations"].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(
                traj["observations"][si : si + self.context_len]
            )
            actions = torch.from_numpy(
                traj["actions"][si : si + self.context_len]
            )
            returns_to_go = torch.from_numpy(
                traj["returns_to_go"][si : si + self.context_len]
            )
            rewards = torch.from_numpy(
                traj["rewards"][si : si + self.context_len]
            )
            timesteps = torch.arange(
                start=si, end=si + self.context_len, step=1
            )

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj["observations"])
            states = torch.cat(
                [
                    states,
                    torch.zeros(
                        ([padding_len] + list(states.shape[1:])),
                        dtype=states.dtype,
                    ),
                ],
                dim=0,
            )

            actions = torch.from_numpy(traj["actions"])
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(
                        ([padding_len] + list(actions.shape[1:])),
                        dtype=actions.dtype,
                    ),
                ],
                dim=0,
            )

            returns_to_go = torch.from_numpy(traj["returns_to_go"])
            returns_to_go = torch.cat(
                [
                    returns_to_go,
                    torch.zeros(
                        ([padding_len] + list(returns_to_go.shape[1:])),
                        dtype=returns_to_go.dtype,
                    ),
                ],
                dim=0,
            )

            rewards = torch.from_numpy(traj["rewards"])
            rewards = torch.cat(
                [
                    rewards,
                    torch.zeros(
                        ([padding_len] + list(rewards.shape[1:])),
                        dtype=rewards.dtype,
                    ),
                ],
                dim=0,
            )

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat(
                [
                    torch.ones(traj_len, dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long),
                ],
                dim=0,
            )

        return (
            timesteps,
            states,
            actions,
            returns_to_go,
            rewards,
            traj_mask,
        )
