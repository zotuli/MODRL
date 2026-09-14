多目标深度强化学习用于并联机器人多目标优化
=======
# Multi-Objective Deep Reinforcement Learning for Parallel-Robot Dimensional Optimization

This repository contains the implementation accompanying the paper **“Multi-Objective Deep Reinforcement Learning for Dimensional Optimization of Parallel Robots.”**( https://ieeexplore.ieee.org/document/11592423 ) The method formulates dimensional design as a sequential decision problem and uses a Gray-code state representation with an attention-enhanced LSTM actor to search for a Pareto set of feasible robot designs.

The code in `rl_for_moo/` is the publication-oriented implementation. The other directories are historical experiments and local research archives; they are not required for reproducing the main experiment.

## Repository layout

```text
rl_for_moo/
├── function_revised.py                         # Robot geometry and objective functions
├── MooEnv_gray.py                               # Gray-code multi-objective environment
├── environments.py                               # Environment/benchmark factory
├── models.py                                    # Actor and critic networks
├── utils.py                                     # Reproducibility, logging, and result helpers
└── train_attention_lstm.py                      # Configurable training entry point

requirements.txt                                 # Python dependencies
```

The original scripts are kept in `rl_for_moo/` for reference. Generated checkpoints, W&B runs, and intermediate research data should remain outside the source distribution. Git ignores these files by default because the complete checkpoint archive is several gigabytes.

## Installation

Python 3.10–3.12 is recommended. Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch wheels are platform- and CUDA-specific. If the generic PyPI wheel is not appropriate for your machine, install the matching PyTorch build from the [official selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements.

## Reproduce the main experiment

Run a short smoke test first:

```bash
python -m rl_for_moo.train_attention_lstm --steps 10 --batch-size 4 --seed 42 --output-dir runs/smoke
```

Run the full experiment with the default settings used by the final script:

```bash
python -m rl_for_moo.train_attention_lstm \
  --steps 80000 \
  --batch-size 256 \
  --seed 42 \
  --output-dir runs/seed_42
```

The default workspace boundary method is PRBD. To run the PABD implementation used in the workspace-method comparison, select it explicitly:

```bash
python -m rl_for_moo.train_attention_lstm \
  --workspace-method PABD \
  --steps 80000 \
  --batch-size 256 \
  --seed 42 \
  --output-dir runs/pabd_seed_42
```

Both methods use the shared environment in `rl_for_moo/MooEnv_gray.py` and the single objective module `rl_for_moo/function_revised.py`. The `--workspace-method` option selects PRBD or PABD inside that module. Always record the selected method with the resulting experiment.

## Simple benchmark environments

The same Gray-code environment can be tested on standard multi-objective problems without running the expensive robot geometry evaluation:

```bash
python -m rl_for_moo.train_attention_lstm --environment zdt2 --steps 1000 --batch-size 64 --output-dir runs/zdt2
python -m rl_for_moo.train_attention_lstm --environment zdt6 --steps 1000 --batch-size 64 --output-dir runs/zdt6
python -m rl_for_moo.train_attention_lstm --environment kursawe --steps 1000 --batch-size 64 --output-dir runs/kursawe
python -m rl_for_moo.train_attention_lstm --environment osyczka_kundu --steps 1000 --batch-size 64 --output-dir runs/osyczka
```

For a custom problem, implement `obj_dim`, `x_dim`, `x_low`, `x_up`, `inital_state`, and `obj(x)`, then register it with `register_problem()` in `rl_for_moo/environments.py`. The current Pareto reward code assumes that all objectives are minimized.

Useful options include `--idx`, `--bits`, `--sequence-length`, `--hidden-size`, and `--hidden-layers`. Checkpoints and the final Pareto archive are written below `--output-dir`.

Weights & Biases logging is optional and disabled by default:

```bash
wandb login
python -m rl_for_moo.train_attention_lstm --wandb --wandb-project parallel-robot-moo
```

No API key is stored in the source code. W&B uses the normal local login and environment-variable configuration.

## Reproducibility notes

The objective values are computed by the robot-specific routines in `rl_for_moo/function_revised.py`. Feasibility constraints and Gray-code transitions are implemented in `rl_for_moo/MooEnv_gray.py`. Results are stochastic; report the seed, device, PyTorch version, and configuration together with every experiment.

The checked-in paper PDF is included for convenience. Before publishing this repository, verify that redistributing the PDF is permitted by the publisher and add the paper's DOI, authors, and citation in this section.

## Data and artifacts

Large `.pth` files and generated experiment logs are intentionally excluded from normal Git commits. For a public release, publish selected weights through a GitHub Release, Git LFS, or an archival repository such as Zenodo, and record the corresponding download URL and checksum here.
