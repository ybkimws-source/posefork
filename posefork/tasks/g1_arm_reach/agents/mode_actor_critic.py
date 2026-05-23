from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class ModeConditionedActorCritic(nn.Module):
    """Actor trunk shared, final action head split by mode one-hot group."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "log",
        **kwargs,
    ):
        if kwargs:
            print(
                "ModeConditionedActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "ModeConditionedActorCritic only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "ModeConditionedActorCritic only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        if "mode" not in obs:
            raise KeyError("Observation group 'mode' is required for ModeConditionedActorCritic.")
        self.num_modes = obs["mode"].shape[-1]

        if len(actor_hidden_dims) < 2:
            raise ValueError("ModeConditionedActorCritic requires actor_hidden_dims of length >= 2 for trunk separation.")
        actor_latent_dim = actor_hidden_dims[-1]
        shared_out_dim = actor_hidden_dims[-2]
        shared_trunk_hidden = actor_hidden_dims[:-2] if len(actor_hidden_dims) > 2 else []
        if len(shared_trunk_hidden) == 0:
            self.shared_trunk = nn.Linear(num_actor_obs, shared_out_dim)
        else:
            self.shared_trunk = MLP(num_actor_obs, shared_out_dim, shared_trunk_hidden, activation)
        activation_cls = nn.ELU if activation == "elu" else nn.ReLU
        self.mode_tails = nn.ModuleList(
            [nn.Sequential(activation_cls(), nn.Linear(shared_out_dim, actor_latent_dim)) for _ in range(self.num_modes)]
        )
        self.actor_heads = nn.ModuleList([nn.Linear(actor_latent_dim, num_actions) for _ in range(self.num_modes)])

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Mode actor shared trunk: {self.shared_trunk}")
        print(f"Mode actor tails: {self.num_modes} x [{activation}, Linear({shared_out_dim}, {actor_latent_dim})]")
        print(f"Mode actor heads: {self.num_modes} x Linear({actor_latent_dim}, {num_actions})")

        self.critics = nn.ModuleList(
            [MLP(num_critic_obs, 1, critic_hidden_dims, activation) for _ in range(self.num_modes)]
        )
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Per-mode critics: {self.num_modes} x {self.critics[0]}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_mean(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        shared = self.shared_trunk(actor_obs)
        mode = obs["mode"]
        if mode.shape[-1] != self.num_modes:
            raise ValueError(f"mode dim mismatch: expected {self.num_modes}, got {mode.shape[-1]}")
        all_means = torch.stack(
            [self.actor_heads[k](self.mode_tails[k](shared)) for k in range(self.num_modes)], dim=1
        )
        return (all_means * mode.unsqueeze(-1)).sum(dim=1)

    def update_distribution(self, obs):
        mean = self._actor_mean(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).clamp(min=1e-6, max=10.0).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self._actor_mean(obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        mode_ids = obs["mode"].argmax(dim=-1)
        values = torch.zeros(critic_obs.shape[0], 1, device=critic_obs.device, dtype=critic_obs.dtype)
        for k in range(self.num_modes):
            mask = mode_ids == k
            if mask.any():
                values[mask] = self.critics[k](critic_obs[mask])
        return values

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
