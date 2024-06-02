import argparse
import os
import random
from datetime import datetime
import time

import d4rl
import gym
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from data import D4RLTrajectoryDataset
from trainer import ReinFormerTrainer
from eval import Reinformer_eval



def experiment(variant):
    # seeding
    seed = variant["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = variant["env"]
    dataset = variant["dataset"]
    
    if dataset == "complete":
        variant["batch_size"] = 16
    if env == "kitchen":
        d4rl_env = f"{env}-{dataset}-v0"
    elif env in ["pen", "door", "hammer", "relocate", "maze2d"]:
        d4rl_env = f"{env}-{dataset}-v1"
    elif env in ["halfcheetah", "hopper", "walker2d", "antmaze"]:
        d4rl_env = f"{env}-{dataset}-v2"
    if env in ["kitchen", "maze2d", "antmaze"]:
        variant["num_eval_ep"] = 100
    if env == "hopper":
        if dataset == "medium" or dataset == "meidum-replay":
            variant["batch_size"] = 256
    
    dataset_path = os.path.join(variant["dataset_dir"], f"{d4rl_env}.pkl")
    device = torch.device(variant["device"])

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    traj_dataset = D4RLTrajectoryDataset(
        env, dataset_path, variant["context_len"], device
    )

    traj_data_loader = DataLoader(
        traj_dataset,
        batch_size=variant["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    data_iter = iter(traj_data_loader)

    state_mean, state_std = traj_dataset.get_state_stats()

    env = gym.make(d4rl_env)
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model_type = variant["model_type"]

    if model_type == "reinformer":
        Trainer = ReinFormerTrainer(
            state_dim=state_dim,
            act_dim=act_dim,
            device=device,
            variant=variant
        )
        def evaluator(model):
            return_mean, _, _, _ = Reinformer_eval(
                model=model,
                device=device,
                context_len=variant["context_len"],
                env = env,
                state_mean=state_mean,
                state_std=state_std,
                num_eval_ep=variant["num_eval_ep"],
                max_test_ep_len=variant["max_eval_ep_len"]
            )
            return env.get_normalized_score(
                return_mean
            ) * 100

    max_train_iters = variant["max_train_iters"]
    num_updates_per_iter = variant["num_updates_per_iter"]
    normalized_d4rl_score_list = []
    for _ in range(1, max_train_iters+1):
        t1 = time.time()
        for epoch in range(num_updates_per_iter):
            try:
                (
                    timesteps,
                    states,
                    next_states,
                    actions,
                    returns_to_go,
                    rewards,
                    traj_mask,
                ) = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                (
                    timesteps,
                    states,
                    next_states,
                    actions,
                    returns_to_go,
                    rewards,
                    traj_mask,
                ) = next(data_iter)

            loss = Trainer.train_step(
                timesteps=timesteps,
                states=states,
                next_states=next_states,
                actions=actions,
                returns_to_go=returns_to_go,
                rewards=rewards,
                traj_mask=traj_mask
            )
            if args.use_wandb:
                wandb.log(
                    data={
                        "training/loss" : loss,
                    }
                )
        t2 = time.time()
        normalized_d4rl_score = evaluator(
            model=Trainer.model
        )
        t3 = time.time()
        normalized_d4rl_score_list.append(normalized_d4rl_score)
        if args.use_wandb:
            wandb.log(
                data={
                        "training/time" : t2 - t1,
                        "evaluation/score" : normalized_d4rl_score,
                        "evaluation/time": t3 - t2
                    }
            )

    if args.use_wandb:
        wandb.log(
            data={
                "evaluation/max_score" : max(normalized_d4rl_score_list),
                "evaluation/last_score" : normalized_d4rl_score_list[-1]
            }
        )
    print(normalized_d4rl_score_list)
    print("=" * 60)
    print("finished training!")
    end_time = datetime.now().replace(microsecond=0)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("finished training at: " + end_time_str)
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=[ "reinformer"], default="reinformer")
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument("--dataset", type=str, default="medium")
    parser.add_argument("--num_eval_ep", type=int, default=10)
    parser.add_argument("--max_eval_ep_len", type=int, default=1000)
    parser.add_argument("--dataset_dir", type=str, default="data/d4rl_dataset/")
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=256)  
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--grad_norm", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_train_iters", type=int, default=10)
    parser.add_argument("--num_updates_per_iter", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    # use_wandb = False
    parser.add_argument("--use_wandb", action='store_true', default=False)
    args = parser.parse_args()
    
    if args.use_wandb:
        wandb.init(
            name=args.env + "-" + args.dataset,
            project="Reinformer",
            config=vars(args)
        )

    experiment(vars(args))
