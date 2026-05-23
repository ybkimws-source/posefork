# ModePPO — design notes

How and why `ModePPO` differs from standard PPO, with the reasoning behind each
change. This is the long-form companion to the short summary in the README.

Source: `posefork/algorithms/mode_ppo.py`. `ModePPO` subclasses rsl_rl's `PPO`
and keeps the trust-region core untouched (clipped surrogate, GAE, KL-based
learning-rate adaptation). It changes exactly two things, and then adds a few
stability levers to keep those changes from blowing up.

---

## The problem: standard PPO does not treat modes fairly

In this task every environment is permanently assigned one of `K = 3` posture
modes (`mode_id`, one-hot, fed in as an observation). The three modes do **not**
have learning signals of the same magnitude. A mode near the neutral posture is
easy and converges fast, so its advantages become small and quiet. A harder mode
keeps producing large, noisy advantages.

Standard PPO normalizes the advantage **globally** — it pools every sample in
the rollout, subtracts the mean, and divides by the standard deviation. With
several modes mixed together, the high-variance mode dominates that global
denominator. The quiet, already-solved modes get flattened relative to it, the
loud mode hogs the gradient, and the result is the familiar failure: easy modes
get all the optimization, hard modes are starved, and modes collapse onto a
single shared solution. This is exactly the mode-0 / mode-2 collapse seen early
in the project.

The two changes below attack this from two different angles. Change 1 removes
the *signal-magnitude distortion*; Change 2 directly compensates for the
*difficulty gap* that remains.

---

## Change 1 — per-mode advantage normalization

Normalize advantages **within each mode** instead of globally. The relevant part
of `compute_returns`:

```python
# turn OFF the parent's global normalization first
self.storage.compute_returns(last_values, self.gamma, self.lam,
                             normalize_advantage=False)

mode_ids = self.storage.observations["mode"].argmax(dim=-1)
...
for k in range(self._num_modes):
    mask  = flat_mode == k
    count = int(mask.sum().item())
    if count == 0:
        continue
    # ... per-mode diagnostics logged here ...
    if count < self.PER_MODE_COUNT_GUARD:        # = 32
        continue                                  # too few samples to normalize safely
    adv_k   = flat_adv[mask]
    adv_std = adv_k.std().clamp(min=self.ADV_STD_FLOOR)   # = 0.02
    normalized[mask] = (adv_k - adv_k.mean()) / (adv_std + 1e-8)
```

Two things matter here:

1. **The parent's global normalization is explicitly disabled**
   (`normalize_advantage=False`), and `__init__` *raises* if
   `normalize_advantage_per_mini_batch=True` is passed — the per-mode pass is the
   single source of truth for normalization, and a second normalization on top of
   it would silently undo the point.
2. Each mode is normalized by **its own mean and std**, so every mode reasons
   only about "which actions are relatively good *within my own distribution*".
   Differences in raw signal magnitude across modes no longer warp the gradient.
   All three modes get an equal vote.

### The `ADV_STD_FLOOR` and its known asymmetry

`ADV_STD_FLOOR = 0.02` clamps the per-mode std from below. The motivation: if a
mode is solved so well that its advantages collapse toward zero, `adv_k.std()`
also goes to zero, and dividing by it sends the gradient to infinity. The floor
stops the denominator from going below 0.02.

**This is a one-sided guard, and that one-sidedness has bitten us.** The floor
prevents the std *collapsing to 0*, but there is no ceiling on the other side.
In a weak-signal regime the floor can fire and **inflate** small raw advantages
instead of protecting against a blow-up:

- In iter6 around step 1434, `mode_1` had `adv_std = 0.0125`, below the floor of
  `0.02`. The denominator was clamped to `0.02`, which multiplied that mode's
  small raw advantages by roughly 50×.
- The value-function loss exploded (~`10^33`) and training died.

So the floor is a collapse guard (toward 0) with no corresponding explosion
guard (toward ∞). Fixing this asymmetry — e.g. a symmetric clamp, or detecting
"weak signal" rather than blindly flooring — is on the near-term list.

### Minimum-count guard

`PER_MODE_COUNT_GUARD = 32`: if a mode has fewer than 32 samples in the rollout,
its advantages are left untouched (no per-mode normalization). With too few
samples the per-mode mean/std are unreliable, and normalizing on them would inject
noise rather than remove it.

---

## Change 2 — worst-mode aggregation

Standard PPO averages the per-sample surrogate loss over the whole batch. With
modes mixed in, that average can hide a starved mode: if two of three modes are
doing well, the mean looks fine even if the third has been abandoned.

So the loss blends the mean over modes with the **worst** mode. From `update`:

```python
per_sample_surrogate = torch.max(surrogate, surrogate_clipped)   # PPO clip, per sample

alpha = float(self.WORST_MODE_ALPHA)      # = 0.3
if num_modes > 0 and alpha > 0.0:
    mode_ids_batch = obs_batch["mode"].argmax(dim=-1)
    mode_means = [per_sample_surrogate[mode_ids_batch == k].mean()
                  for k in range(num_modes) if (mode_ids_batch == k).any()]
    if len(mode_means) >= 2:
        worst_val = torch.stack(mode_means).max(dim=0).values   # surrogate is a loss → max = worst
        mean_part = per_sample_surrogate.mean()
        surrogate_loss = (1.0 - alpha) * mean_part + alpha * worst_val
    else:
        surrogate_loss = per_sample_surrogate.mean()
else:
    surrogate_loss = per_sample_surrogate.mean()
```

- `per_sample_surrogate` is a *loss* (the negated, clipped PPO surrogate), so the
  **maximum** across mode-means is the worst-performing mode — the one we most
  want to pull up.
- With `alpha = 0.3`: roughly "follow the mean overall, but always spend 30% of
  the objective on whichever mode is currently furthest behind." It behaves like
  a soft-min / CVaR-style term — the hardest mode is guaranteed a slice of the
  gradient every step, so it can't be quietly dropped.
- Edge handling: the worst-mode term only kicks in when at least two modes are
  present in the batch; otherwise it falls back to the plain mean.

Tuning intuition: `alpha = 0` recovers standard PPO (mean only); `alpha → 1`
chases only the worst mode and destabilizes overall performance. `0.3` is the
compromise that held in practice.

Where Change 1 equalizes the *gradient scale* across modes, Change 2 equalizes
*attention* — they target the same goal (mode fairness) at two different layers.

---

## Why those two weren't enough — stability levers

Changes 1 and 2 produce mode *fairness* but not numerical *stability*. Per-mode
heads, the one-sided floor, and the worst-mode term together make the gradient
rougher, so three more knobs were tightened (see the 05-21 / 05-22 logs):

| Lever | Value | Was | Why |
| --- | --- | --- | --- |
| `ADV_STD_FLOOR` | `0.02` | — | divide-by-zero guard for per-mode std (but one-sided — see above) |
| `desired_kl` | `0.02` | `0.01` | loosened — in multi-mode, one mode's large update spikes KL and over-shrinks the shared learning rate |
| `max_grad_norm` | `0.3` | `1.0` | tightened — so a spike (e.g. from the floor over-amplifying) gets clipped before it kills a step |

With these three together, iter5_stab reached a `mode_span ≈ 0.387` plateau
without diverging. (Other runs report figures like `0.347` / `0.40`; the number
moves per run/iter, which is why the README states it qualitatively rather than
pinning one value.)

---

## Summary

`ModePPO` is not a new algorithm. It is rsl_rl PPO with:

1. advantage normalization moved from global to **per-mode**, and
2. the surrogate loss blended with the **worst mode** (`α = 0.3`),

plus three stability levers to absorb the resulting roughness. One of those
levers — the `ADV_STD_FLOOR` — is currently a known, asymmetric weak point and is
slated for a fix.

Companion classes: `ModeConditionedActorCritic` (per-mode policy/value heads),
defined in `posefork/tasks/g1_arm_reach/agents/mode_actor_critic.py`.
