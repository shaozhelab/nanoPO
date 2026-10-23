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
| `--wandb` | log scores and losses to Weights & Biases (run `wandb login` once first) |
| `--video` | record one episode of the final policy |

Hyperparameters default to the per-environment values in `DEFAULTS` at the top of `ppo/ppo.py`
(HalfCheetah uses the MuJoCo settings from Table 3 of the PPO paper). Any of them can be overridden, e.g.

```bash
python -m ppo.ppo --env-id CartPole-v1 --seed 1 --lr 1e-4 --num-batches 100
```

Each run writes to `results/`:

| File | Content |
|---|---|
| `{env}-seed{seed}.png` | mean score per batch |
| `{env}-seed{seed}.pt` | trained networks |
| `{env}-seed{seed}.mp4` | one episode of the final policy (with `--video`) |

## Results

5 seeds per environment.

- **CartPole-v1** (300 batches × 512 = 153,600 steps): all 5 seeds score the maximum 500 in every batch from batch 133 on.
- **HalfCheetah-v5** (489 batches × 2,048 = 1,001,472 steps): mean score over the last 20 batches is
  1,411.8 / 1,363.0 / 1,432.3 / 1,391.9 / 3,707.2 for seeds 1–5.
