from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain

from rsl_rl.algorithms import PPO


class ModePPO(PPO):
    """PPO with per-mode independent advantage normalization and per-mode diagnostics."""

    PER_MODE_COUNT_GUARD = 32
    WORST_MODE_ALPHA = 0.3
    ADV_STD_FLOOR = 0.02  # minimum per-mode adv std; prevents 1/std amplification when signal is weak

    def __init__(self, *args, **kwargs):
        if kwargs.get("normalize_advantage_per_mini_batch", False):
            raise ValueError(
                "ModePPO performs per-mode normalization in compute_returns(); "
                "normalize_advantage_per_mini_batch must be False."
            )
        super().__init__(*args, **kwargs)
        self._rollout_log: dict[str, float] = {}

    def compute_returns(self, obs):
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantage=False)

        mode_onehot = self.storage.observations["mode"]
        mode_ids = mode_onehot.argmax(dim=-1)
        self._num_modes = mode_onehot.shape[-1]

        advantages = self.storage.advantages
        returns = self.storage.returns
        values = self.storage.values

        flat_adv = advantages.view(-1)
        flat_ret = returns.view(-1)
        flat_val = values.view(-1)
        flat_mode = mode_ids.view(-1)

        normalized = flat_adv.clone()
        self._rollout_log = {}
        for k in range(self._num_modes):
            mask = flat_mode == k
            count = int(mask.sum().item())
            self._rollout_log[f"mode_{k}/count"] = float(count)
            if count == 0:
                continue
            adv_k = flat_adv[mask]
            self._rollout_log[f"mode_{k}/value_mean"] = flat_val[mask].mean().item()
            self._rollout_log[f"mode_{k}/return_mean"] = flat_ret[mask].mean().item()
            self._rollout_log[f"mode_{k}/advantage_mean"] = adv_k.mean().item()
            self._rollout_log[f"mode_{k}/advantage_std"] = adv_k.std().item() if count > 1 else 0.0
            if count < self.PER_MODE_COUNT_GUARD:
                continue
            adv_std = adv_k.std().clamp(min=self.ADV_STD_FLOOR)
            normalized[mask] = (adv_k - adv_k.mean()) / (adv_std + 1e-8)

        self.storage.advantages = normalized.view_as(advantages)

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        num_modes = getattr(self, "_num_modes", 0)
        mode_clip_hits = torch.zeros(num_modes, device=self.device)
        mode_clip_total = torch.zeros(num_modes, device=self.device)
        mode_surr_sum = torch.zeros(num_modes, device=self.device)
        mode_surr_count = torch.zeros(num_modes, device=self.device)
        mode_worst_picks = torch.zeros(num_modes, device=self.device)

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                from rsl_rl.utils import string_to_callable  # noqa: F401
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_clipped = -torch.squeeze(advantages_batch) * ratio_clipped
            per_sample_surrogate = torch.max(surrogate, surrogate_clipped)

            alpha = float(self.WORST_MODE_ALPHA)
            if num_modes > 0 and alpha > 0.0:
                mode_ids_batch_full = obs_batch["mode"].argmax(dim=-1)
                mode_means = []
                for k in range(num_modes):
                    m = mode_ids_batch_full == k
                    if m.any():
                        mode_means.append(per_sample_surrogate[m].mean())
                if len(mode_means) >= 2:
                    stacked = torch.stack(mode_means)
                    worst_val, worst_idx = stacked.max(dim=0)
                    mean_part = per_sample_surrogate.mean()
                    surrogate_loss = (1.0 - alpha) * mean_part + alpha * worst_val
                    mode_worst_picks[worst_idx] += 1
                else:
                    surrogate_loss = per_sample_surrogate.mean()
            else:
                surrogate_loss = per_sample_surrogate.mean()

            if num_modes > 0:
                clipped_mask = (ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)
                mode_ids_batch = obs_batch["mode"][:original_batch_size].argmax(dim=-1)
                clipped_mask = clipped_mask[:original_batch_size].float()
                per_sample_orig = per_sample_surrogate[:original_batch_size].detach()
                for k in range(num_modes):
                    m = mode_ids_batch == k
                    mode_clip_total[k] += m.sum()
                    mode_clip_hits[k] += clipped_mask[m].sum()
                    if m.any():
                        mode_surr_sum[k] += per_sample_orig[m].sum()
                        mode_surr_count[k] += m.sum()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        total_worst_picks = mode_worst_picks.sum().item()
        for k in range(num_modes):
            total = mode_clip_total[k].item()
            if total > 0:
                loss_dict[f"mode_{k}/clip_fraction"] = (mode_clip_hits[k].item() / total)
            else:
                loss_dict[f"mode_{k}/clip_fraction"] = 0.0
            cnt = mode_surr_count[k].item()
            if cnt > 0:
                loss_dict[f"mode_{k}/surrogate_mean"] = (mode_surr_sum[k].item() / cnt)
            else:
                loss_dict[f"mode_{k}/surrogate_mean"] = 0.0
            if total_worst_picks > 0:
                loss_dict[f"mode_{k}/worst_pick_rate"] = (mode_worst_picks[k].item() / total_worst_picks)
            else:
                loss_dict[f"mode_{k}/worst_pick_rate"] = 0.0
        loss_dict["worst_mode_alpha"] = float(self.WORST_MODE_ALPHA)
        loss_dict.update(self._rollout_log)

        return loss_dict
