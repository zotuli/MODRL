# RL for multi-objective optimization

This directory contains the main training code for the published experiment.

- `function_revised.py` contains the parallel-robot geometry, kinematics, constraints, and objective calculations.
- `MooEnv_gray.py` implements Gray-code state transitions, feasibility handling, Pareto-archive maintenance, and rewards.
- `models.py` contains the attention-LSTM actor and MLP critic.
- `utils.py` contains seeding, Polyak updates, result export, and optional W&B logging.
- `train_attention_lstm.py` is the recommended command-line entry point.
- `environments.py` selects the parallel-robot backend or simple benchmark problems.
- `function_revised.py` contains both PRBD and PABD objective implementations, selected by the workspace method.

From the repository root, run:

```bash
python -m rl_for_moo.train_attention_lstm --seed 42 --output-dir runs/seed_42
```

PRBD is the default. Switch to PABD with:

```bash
python -m rl_for_moo.train_attention_lstm --workspace-method PABD --seed 42 --output-dir runs/pabd_seed_42
```

Simple benchmark example:

```bash
python -m rl_for_moo.train_attention_lstm --environment zdt2 --steps 1000 --batch-size 64 --output-dir runs/zdt2
```

The two files named `Moo_RL_AC_modified_attention_newlstm*.py` are retained as historical paper scripts. New experiments should use the configurable entry point above so that output paths and hyperparameters are recorded in `config.json`.
