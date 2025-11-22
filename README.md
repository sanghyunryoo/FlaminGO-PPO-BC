# Isaac LAB for Flamingo

## Brief

A compact fork adding a **rough-terrain PPO–BC framework** for a **two-wheel biped robot** (locomotion on challenging terrains).

> **Origin**  
> Modified from **[jaykorea/Isaac-RL-Two-wheel-Legged-Bot](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot)**.  
> This repository focuses on integrating **PPO + Behavior Cloning** for robust off-road locomotion, with light refactors to training / evaluation scripts.

---

## Usage

### 1. Train point-foot (PF) model

```bash
python scripts/co_rl/train.py \
  --task Isaac-Velocity-Rough-FlamingoPF-v1-ppo \
  --num_envs 4096 \
  --headless \
  --algo ppo \
  --num_policy_stacks 2 \
  --num_critic_stacks 2

### 2. Train wheel version with PPO-BC

```bash
python scripts/co_rl/train.py \
  --task Isaac-Velocity-Rough-BC-Flamingo-v1-ppo \
  --num_envs 4096 \
  --headless \
  --algo ppo \
  --num_policy_stacks 2 \
  --num_critic_stacks 2 \
  --bc {pf weight path}

### Example
```bash
python scripts/co_rl/train.py \
  --task Isaac-Velocity-Rough-BC-Flamingo-v1-ppo \
  --num_envs 4096 \
  --headless \
  --algo ppo \
  --num_policy_stacks 2 \
  --num_critic_stacks 2 \
  --bc 2025-02-10_02-31-01
