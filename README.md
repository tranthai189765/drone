# DRONEs: Deep Reinforcement Optimization for Network *k*-Connectivity Restoration in UAV Swarms

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="Gymnasium" src="https://img.shields.io/badge/Gymnasium-1.2-0081A5"></a>
  <a href="#"><img alt="Stable-Baselines3" src="https://img.shields.io/badge/SB3-2.7-1f6feb"></a>
  <a href="#"><img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange"></a>
</p>

A scalable **Deep Reinforcement Learning (DRL)** framework for restoring **k‑vertex‑connectivity** in
Unmanned Aerial Vehicle (UAV) swarms while **minimizing total displacement**. A single policy with a
**permutation‑invariant, attention‑based encoder** is trained on small swarms (5–50 UAVs) and generalizes
**zero‑shot** to much larger fleets (e.g., 100 UAVs) without retraining, delivering real‑time, feed‑forward
relocation decisions.

> This repository accompanies the paper *"DRONEs: Deep Reinforcement Optimization for Network k‑Connectivity
> Restoration Enhancement in UAVs."* See [Citation](#citation).

---

## Table of Contents

- [Motivation](#motivation)
- [Key Features](#key-features)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The `KRes` Environment](#the-kres-environment)
- [Training Configuration](#training-configuration)
- [Baseline: Node Converging (NC)](#baseline-node-converging-nc)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)

---

## Motivation

In time‑critical missions (disaster response, emergency communications, search‑and‑rescue), multi‑hop
communication among drones must stay reliable under node failures and mobility. A principled measure of this
robustness is **k‑vertex connectivity**: a network is *k*-connected if every pair of nodes is linked by at
least *k* internally disjoint paths, implying resilience to up to *k − 1* node failures (Menger's theorem).

The **k‑connectivity restoration problem** asks: after damage, how should UAVs relocate — with **minimum total
movement** — so that the communication graph becomes *k*-connected again? This is closely related to the
minimum *k*-vertex‑connected spanning subgraph problem, which is **NP‑hard for k ≥ 2**.

Classical approaches (mixed‑integer programming, hand‑crafted mobility heuristics) yield strong baselines on
small instances but **scale poorly** with swarm size and operating area, since each relocation triggers global
connectivity checks. This project replaces repeated combinatorial solving with a **learned policy** that
amortizes the optimization and runs in near real‑time at inference.

## Key Features

- 🧩 **Permutation‑invariant & size‑agnostic policy** — a Set‑Transformer–style encoder (Self‑Attention `SAB` /
  Induced Set‑Attention `ISAB`) processes a variable number of UAVs with no architectural changes.
- 🚀 **Zero‑shot scale generalization** — trained on small swarms, deployed on large ones without retraining.
- 🧠 **CTDE architecture** — *Centralized Training with Decentralized Execution*: a shared backbone, a
  **per‑agent actor head**, and a **pooled global critic**.
- 🎯 **Hungarian post‑assignment refinement** — optimally re‑maps the predicted goal set to vehicles in
  `O(N³)`, cutting total displacement **without altering the learned topology**.
- 📚 **Two‑phase curriculum** — Phase 1 uses a feasibility‑oriented reward (drive toward the swarm center),
  Phase 2 switches to the movement‑minimizing reward.
- 🔁 **Multiple DRL algorithms** — PPO (primary), plus A2C and DQN explorations.
- ⚡ **Linear / scaled attention** for fast feature extraction at scale.
- 🖼️ **PyGame rendering** of the grid world for visualization.

## Method Overview

The task is cast as a finite‑horizon **Markov Decision Process** `M = (S, A, P, r, γ)`:

| Component | Description |
|-----------|-------------|
| **State** `sₜ` | Current and initial UAV positions plus per‑agent features (centers, movement scale, degree, node‑connectivity cues). |
| **Action** `aₜ` | Joint move; each UAV picks one of **9 discrete grid actions** (8 compass directions + `STAY`), executed in parallel. |
| **Transition** `P` | Apply kinematics, then rebuild the communication graph `G(Xₜ₊₁)`. |
| **Reward** `rₜ` | Trades connectivity progress against motion cost; a connectivity bonus activates only when `κ(G) ≥ k`, with a movement penalty and a failure penalty beyond the horizon `τ = 2S`. |

**Pipeline.** Each UAV is encoded as a normalized feature vector → a row‑wise **Encoder MLP** → stacked
**attention feature‑extractor blocks** (Multi‑Head Attention + Feed‑Forward) → per‑agent embeddings. These feed
two heads: a **per‑agent action head** (actor) and a **pooled global critic**. When the policy emits an
unordered set of target positions, the **Hungarian algorithm** assigns drones to targets to minimize total
travel.

```
Xₜ ∈ ℝ^{N×f}  ──▶  Encoder MLP  ──▶  [ MHA + FFN ] × L  ──▶  H ∈ ℝ^{N×d}
                                                              ├──▶ per‑agent Actor head  ─▶ actions
                                                              └──▶ Pool ─▶ global Critic  ─▶ value
                                                                              │
                                       (unordered goal set) ────────────────▶ Hungarian re‑assignment ─▶ targets
```

## Repository Structure

```
SOICT2025/
├── main.py                     # Entry point: train a PPO attention policy, then roll out
├── dqn.ipynb                   # DQN experiments / analysis notebook
├── pyproject.toml              # Project metadata & dependencies (uv-managed)
├── requirements.txt            # Pinned dependencies (pip)
├── uv.lock                     # uv lockfile
│
├── KResRL/                     # Core library: k-connectivity Restoration via RL
│   ├── ppo.py                  # PPO training wrapper (EnvOptions / RLOptions / TrainOptions)
│   ├── utils.py
│   ├── environment/
│   │   ├── env.py              # `KRes` Gymnasium environment (grid, graph, reward, Hungarian matching)
│   │   ├── utils.py
│   │   └── view.py             # PyGame grid renderer (GridView)
│   └── policy/
│       ├── ac_policy.py        # NodeLevelActorCriticPolicy + per-agent ActionNet / pooled ValueNet
│       ├── att_base.py         # Attention blocks: MAB, SAB, ISAB, and a Mixture-of-Experts FFN
│       ├── att_policy.py       # AttFeaturesExtractor (Set-Transformer encoder)
│       ├── graph_base.py       # GCN building blocks
│       └── graph_policy.py     # Graph-based feature extractor (GCN alternative)
│
├── gym_agent/                  # Lightweight in-house agent framework
│   ├── agent/                  # agent_base, on_policy, buffers, callbacks
│   ├── core/                   # policy, transforms
│   └── utils.py
│
└── NC/
    └── main.py                 # Node Converging (NC) heuristic baseline solver
```

Key files: [main.py](main.py), [KResRL/environment/env.py](KResRL/environment/env.py),
[KResRL/policy/att_base.py](KResRL/policy/att_base.py), [KResRL/ppo.py](KResRL/ppo.py),
[NC/main.py](NC/main.py).

## Installation

Requires **Python 3.13+**.

### Option A — [uv](https://docs.astral.sh/uv/) (recommended)

```bash
# from the repository root
uv sync
```

### Option B — pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Core dependencies: `torch`, `torch-geometric`, `stable-baselines3`, `gymnasium`, `networkx`, `munkres`
(Hungarian algorithm), `numpy`, `scipy`, `pygame`.

## Quick Start

Train a PPO policy with the attention encoder and roll it out with rendering:

```bash
# with uv
uv run main.py

# or, with an activated venv
python main.py
```

This (see [main.py](main.py)):

1. builds a vectorized `KRes` environment,
2. trains a PPO policy using a `SAB` (Self‑Attention Block) feature extractor,
3. saves the model to `trained_model.zip`, and
4. runs a rendered evaluation episode.

## The `KRes` Environment

`KRes` ([KResRL/environment/env.py](KResRL/environment/env.py)) is a Gymnasium environment modeling a UAV swarm
on an `S × S` grid.

```python
from KResRL.environment.env import KRes

env = KRes(
    n_drones=5,
    k=2,                       # target vertex-connectivity
    min_size=4, max_size=4,    # grid side is resampled in [min_size, max_size] each reset
    return_state="features",   # "features" | "grid" | "pos"
    return_reward="global",    # "global" | "sparse" | "hybrid"
    normalize_features=True,
    render_mode="human",       # or "rgb_array" / None
    render_fps=1,
)

obs, info = env.reset()
action = env.action_space.sample()          # MultiDiscrete([9] * n_drones)
obs, reward, done, truncated, info = env.step(action)
```

**Observation (`features` mode).** An `(N, N + F)` matrix: an `N × N` adjacency block (optional) concatenated
with `F` per‑agent node features, all normalized to `[0, 1]`. The node features include current position
(`row`, `col`), formation center, initial formation center, movement scale (`1/size`), accumulated movement,
node `degree`, and local `node_connectivity` cues.

**Action.** `MultiDiscrete([9] * n_drones)` — each drone selects one of:

```
0:STAY  1:Up  2:Up-Right  3:Right  4:Down-Right  5:Down  6:Down-Left  7:Left  8:Up-Left
```

**Termination.** An episode ends when the minimum node‑connectivity reaches `k` (success) or the step limit is
hit. The environment also computes an optimal drone‑to‑target assignment via the **Hungarian (Munkres)**
algorithm to report the minimal‑displacement matching.

## Training Configuration

Training is driven by three dataclasses in [KResRL/ppo.py](KResRL/ppo.py):

```python
from KResRL.ppo import EnvOptions, RLOptions, TrainOptions, train
from KResRL.policy.att_policy import SAB, AttFeatureOptions, AttPolicyOptions

env_options = EnvOptions(n_envs=5, n_drones=5, k=2, size=4, alpha=0.001)

policy_options = AttPolicyOptions(
    features_extractor_kwargs=AttFeatureOptions(NNBlock=SAB, n_layers=6),
    net_arch={"pi": [256, 128], "vf": [256, 128]},
)

rl_options = RLOptions(gamma=0.99, n_steps=256)
train_options = TrainOptions(total_timesteps=100_000, log_interval=100)

model = train(env_options, policy_options, rl_options, train_options)
model.save("trained_model")
```

| Option group | Notable fields |
|--------------|----------------|
| `EnvOptions` | `n_envs`, `n_drones`, `k`, `size`, `alpha`, `return_reward` |
| `RLOptions`  | `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda` |
| `TrainOptions` | `total_timesteps`, `log_interval`, `progress_bar`, `tb_log_name` |
| `AttFeatureOptions` | `NNBlock` (`SAB`/`ISAB`), `n_layers`, `embed_dim`, `num_heads`, `d_ff`, `dropout`, optional Mixture‑of‑Experts |

You can swap the attention encoder for a **GCN‑based** extractor via `GraphPolicyOptions` /
`GraphFeatureOptions` (see [KResRL/policy/graph_policy.py](KResRL/policy/graph_policy.py)).

## Baseline: Node Converging (NC)

[NC/main.py](NC/main.py) implements the **Node Converging** heuristic used as the reference baseline. It
iteratively moves the drone furthest from the swarm center one step inward until the graph becomes
*k*-connected, accumulating the total movement cost. Because it relocates **one drone at a time** with global
connectivity checks, its runtime grows quickly with swarm size, connectivity requirement, and grid area.

```bash
python NC/main.py
```

The framework also evaluates **hybrid DRL+NC** variants, which run the learned policy until ~80% of drones
converge, then apply NC for local refinement.

## Results

Reported metric: **total movement** = sum of Euclidean distances traveled by all drones (lower is better),
averaged over 50 randomized trials. Defaults `(n, k, S) = (50, 3, 100)` unless varied.

**Total movement vs. number of drones** (`k = 3`, `S = 100`):

| Method  | n = 25 | n = 50 | n = 100 |
|---------|:------:|:------:|:-------:|
| NC      | **966.12** | 1960.74 | 4060.62 |
| DQN     | 1043.54 | 1923.18 | 3957.92 |
| A2C     | 1048.38 | 1927.13 | 3939.84 |
| PPO     | 1044.83 | 1913.16 | 3918.83 |
| DQN+NC  | 1023.08 | 1904.46 | 3884.16 |
| A2C+NC  | 1023.47 | 1898.66 | 3874.18 |
| **PPO+NC** | 1019.16 | **1886.48** | **3848.09** |

**Training strategies & Hungarian refinement** (PPO, Small/Medium/Large):

| Variant | Small | Medium | Large |
|---------|:-----:|:------:|:-----:|
| Progressive single‑phase | 528.47 | 1944.73 | 5902.11 |
| Progressive two‑phase | 526.01 | 1913.31 | 5507.25 |
| Progressive two‑phase **+ Hungarian** | **525.61** | **1886.48** | **5357.45** |

**Takeaways.**

- On **small** swarms/grids (e.g., `n = 25`, `S = 50`), the NC heuristic is hard to beat.
- As the swarm or arena **grows**, DRL — and especially **PPO+NC** — consistently achieves the lowest total
  movement while running far faster than NC (it updates all drones in parallel).
- The **two‑phase curriculum** reduces movement by up to **6.7%** over single‑phase training at large scale;
  **Hungarian** post‑assignment adds a further **~2.7%** improvement on large instances.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{minh2025drones,
  title     = {DRONEs: Deep Reinforcement Optimization for Network k-Connectivity Restoration Enhancement in UAVs},
  author    = {Trinh, The Minh and Tran, Quoc Thai and Nguyen, Van Son and Nguyen, Nam Phuong and Nguyen, Thi Hanh},
  year      = {2025},
  note      = {Hanoi University of Science and Technology; Phenikaa University}
}
```

**Authors:** Trinh The Minh¹, Tran Quoc Thai¹, Nguyen Van Son³, Nguyen Nam Phuong¹, Nguyen Thi Hanh²⁴ (corresponding).
¹ Hanoi University of Science and Technology · ²·³·⁴ Phenikaa University, Hanoi, Vietnam.

## Acknowledgment

This research is funded by **Phenikaa University** under grant number **PU2024‑2‑A‑05**.
