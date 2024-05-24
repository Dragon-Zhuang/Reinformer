import torch
import torch.nn.functional as F
from model import ReinFormer
from lamb import Lamb



class ReinFormerTrainer:
    def __init__(
        self, 
        state_dim,
        act_dim,
        device,
        variant
    ):
        super().__init__()
                
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = device
        self.grad_norm = variant["grad_norm"]

        self.model = ReinFormer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=variant["n_blocks"],
            h_dim=variant["embed_dim"],
            context_len=variant["context_len"],
            n_heads=variant["n_heads"],
            drop_p=variant["dropout_p"],
            init_temperature=variant["init_temperature"],
            target_entropy=-self.act_dim
        ).to(self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["lr"],
            weight_decay=variant["wd"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/variant["warmup_steps"], 1)
        )

        self.tau = variant["tau"]
        self.context_len=variant["context_len"]


        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

    
    def train_step(
        self,
        timesteps,
        states,
        next_states,
        actions,
        returns_to_go,
        rewards,
        traj_mask,
    ):
        self.model.train()
        # data to gpu ------------------------------------------------
        timesteps = timesteps.to(self.device)      # B x T
        states = states.to(self.device)            # B x T x state_dim
        next_states = next_states.to(self.device)  # B x T x state_dim
        actions = actions.to(self.device)          # B x T x act_dim
        returns_to_go = returns_to_go.to(self.device).unsqueeze(
            dim=-1
        )                                          # B x T x 1
        
        rewards = rewards.to(self.device).unsqueeze(
            dim=-1
        )                                          # B x T x 1
        traj_mask = traj_mask.to(self.device)      # B x T

        # model forward ----------------------------------------------
        (
            returns_to_go_preds,
            actions_dist_preds,
            _,
        ) = self.model.forward(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
        )

        returns_to_go_target = torch.clone(returns_to_go).view(
            -1, 1
        )[
            traj_mask.view(-1,) > 0
        ]
        returns_to_go_preds = returns_to_go_preds.view(-1, 1)[
            traj_mask.view(-1,) > 0
        ]

        # returns_to_go_loss -----------------------------------------
        norm = returns_to_go_target.abs().mean()
        u = (returns_to_go_target - returns_to_go_preds) / norm
        returns_to_go_loss = torch.mean(
            torch.abs(
                self.tau - (u < 0).float()
            ) * u ** 2
        )
        # action_loss ------------------------------------------------
        actions_target = torch.clone(actions)
        log_likelihood = actions_dist_preds.log_prob(
            actions_target
            ).sum(axis=2)[
            traj_mask > 0
        ].mean()
        entropy = actions_dist_preds.entropy().sum(axis=2).mean()
        action_loss = -(log_likelihood + self.model.temperature().detach() * entropy)

        loss = returns_to_go_loss + action_loss

        # optimization -----------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.grad_norm
        )
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        self.scheduler.step()

        return loss.detach().cpu().item()