# nanoPO

Minimal, readable implementations of the policy-optimization ("PO") algorithms behind
LLM post-training, each written from its paper.

These algorithms share one loop: sample from the current policy, score the samples, estimate
how much better each sample was than expected (the advantage), and update the policy without
moving it too far. What changes from one algorithm to the next is how each step is done.
nanoPO first gets this loop right in classic RL, where every piece can be checked against
known results, then carries it to language models:

- **PPO on control tasks** (this repo today): a learned value network as the baseline, GAE
  for the advantage, and a clipped probability ratio to bound each update. Verified on
  CartPole and HalfCheetah.
- **PPO for LLMs**: an episode is a generated response and each token is an action; a reward
  model or verifier scores the full response, and a KL penalty keeps the policy close to the
  reference model.
- **GRPO**: removes the value network and uses the mean score of a group of responses to the
  same prompt as the baseline.

## Install

Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch "gymnasium[classic-control,mujoco]" numpy matplotlib wandb moviepy
```

Tested with torch 2.13.0, gymnasium 1.4.0, mujoco 3.17.0, numpy 2.5.4, matplotlib 3.11.2, wandb 0.31.0, moviepy 2.2.1.

## Run

From the repo root:

```bash
python -m ppo.ppo --env-id CartPole-v1 --seed 1
python -m ppo.ppo --env-id HalfCheetah-v5 --seed 1
```

| Flag | Effect |
|---|---|
| `--exp` | experiment from `EXPERIMENTS` in `ppo/ppo.py` (default `ppo`, the full algorithm) |
| `--wandb` | log scores, losses, approx KL and clip fraction to Weights & Biases (run `wandb login` once first) |
| `--video` | record one episode of the final policy |

Hyperparameters default to the per-environment values in `DEFAULTS` at the top of `ppo/ppo.py`
(HalfCheetah uses the MuJoCo settings from Table 3 of the PPO paper). Any of them can be overridden, e.g.

```bash
python -m ppo.ppo --env-id CartPole-v1 --seed 1 --lr 1e-4 --num-batches 100
```

Each experiment changes a few settings from the defaults, e.g. `no-clip` or `reinforce`
(one gradient step per batch, which makes the update REINFORCE with a value baseline):

```bash
python -m ppo.ppo --env-id HalfCheetah-v5 --seed 1 --exp no-clip
```

Each run writes to `results/{env}/{exp}/`:

| File | Content |
|---|---|
| `seed{seed}.csv` | per-batch score, losses, approx KL and clip fraction |
| `seed{seed}.png` | mean score per batch |
| `seed{seed}.pt` | trained networks |
| `seed{seed}.mp4` | one episode of the final policy (with `--video`) |

## Results

8 seeds per configuration. Final score is the mean episode score over the last 20 batches.

- **CartPole-v1** (300 batches × 512 steps): all 8 seeds reach the maximum score of 500 and hold it for the rest of training.
- **HalfCheetah-v5** (489 batches × 2048 steps): median final score 1401.9, on par with the 1442.64 reported by CleanRL.

| Configuration | Final score (median, HalfCheetah-v5) |
|---|---:|
| PPO | 1401.9 |
| 1 epoch | 942.5 |
| No advantage normalization | 772.9 |
| No GAE (λ = 1) | 630.3 |
| No clipping | 137.7 |
| REINFORCE | −120.8 |
