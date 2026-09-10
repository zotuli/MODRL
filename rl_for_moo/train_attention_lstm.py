"""Command-line training entry point for the publication experiment.

Examples
--------
    python -m rl_for_moo.train_attention_lstm --steps 10 --batch-size 4
    python -m rl_for_moo.train_attention_lstm --seed 42 --output-dir runs/seed_42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from .environments import build_environment
from .models import AttentionLSTMActor, MLP
from .utils import ExperimentLogger, elite_list_to_dataframe, set_seed, soft_update


def save_results(
    output_dir: Path,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    elite_list: list[tuple[np.ndarray, np.ndarray]],
    problem: object,
    step: int,
) -> None:
    """Persist model weights and the current Pareto archive."""

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), output_dir / f"actor_step_{step}.pth")
    torch.save(critic.state_dict(), output_dir / f"critic_step_{step}.pth")
    elite_list_to_dataframe(elite_list, problem.obj_dim, problem.x_dim).to_csv(
        output_dir / f"elite_step_{step}.csv", index=False
    )


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env, problem = build_environment(
        args.environment,
        args.idx,
        args.bits,
        args.batch_size,
        args.workspace_method,
    )
    states = env.reset(seed=args.seed)
    state_size = states.shape[1]
    hidden_size = args.hidden_size or state_size

    actor = AttentionLSTMActor(
        input_size=state_size,
        output_size=state_size,
        hidden_size=hidden_size,
        num_layers=args.hidden_layers,
    ).to(device)
    critic = MLP(
        input_size=state_size * 2,
        output_size=1,
        hidden_size=args.critic_hidden_size,
        num_layers=args.hidden_layers,
    ).to(device)
    target_actor = AttentionLSTMActor(
        state_size, state_size, hidden_size, args.hidden_layers
    ).to(device)
    target_critic = MLP(
        state_size * 2, 1, args.critic_hidden_size, args.hidden_layers
    ).to(device)

    # Preserve the initialization used by the original final script.
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
    logger = ExperimentLogger(args.wandb, args.wandb_project, args.wandb_name)

    history = torch.zeros(
        args.batch_size, args.sequence_length, state_size, device=device
    )
    history[:, -1, :] = torch.as_tensor(states, dtype=torch.float32, device=device)
    states_current = history[:, -1, :]
    total_reward = 0.0

    try:
        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            action_probabilities = actor(history)
            actions = torch.distributions.Bernoulli(
                probs=action_probabilities
            ).sample()
            actions = env.select_action(actions.cpu().numpy())

            next_states, rewards = env.step(actions)
            next_states_tensor = torch.as_tensor(
                next_states, dtype=torch.float32, device=device
            )
            rewards_tensor = torch.as_tensor(
                rewards, dtype=torch.float32, device=device
            ).unsqueeze(1)
            total_reward += float(rewards_tensor.mean().item())

            next_history = torch.cat(
                (history[:, 1:, :], next_states_tensor.unsqueeze(1)), dim=1
            )
            with torch.no_grad():
                next_actions = target_actor(next_history)
                next_q = target_critic(
                    torch.cat((next_states_tensor, next_actions), dim=1)
                )
                target = rewards_tensor + args.gamma * next_q

            # These two lines intentionally mirror the final paper script.
            critic_q = critic(torch.cat((states_current, next_actions), dim=1))
            critic_loss = nn.MSELoss()(critic_q, target)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            predicted_actions = actor(history)
            actor_q = critic(torch.cat((states_current, predicted_actions), dim=1))
            actor_loss = -actor_q.mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            soft_update(target_actor, actor, args.tau)
            soft_update(target_critic, critic, args.tau)
            states_current = next_states_tensor
            history = next_history

            logger.log(
                {
                    "step": step,
                    "reward": total_reward,
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "elite_count": len(env.elite_list),
                }
            )

            if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                save_results(output_dir, actor, critic, env.elite_list, problem, step)
                print(
                    f"step={step:>6} reward={total_reward:.4f} "
                    f"elites={len(env.elite_list)} "
                    f"time={time.perf_counter() - step_start:.3f}s"
                )
    finally:
        final_step = args.steps
        save_results(output_dir, actor, critic, env.elite_list, problem, final_step)
        torch.save(actor.state_dict(), output_dir / "final_actor.pth")
        torch.save(critic.state_dict(), output_dir / "final_critic.pth")
        elite_list_to_dataframe(
            env.elite_list, problem.obj_dim, problem.x_dim
        ).to_csv(output_dir / "elite_final.csv", index=False)
        logger.close()

    metadata = vars(args).copy()
    metadata.update(
        {
            "device": str(device),
            "state_size": state_size,
            "objective_dimension": problem.obj_dim,
            "decision_dimension": problem.x_dim,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idx", type=int, default=1)
    parser.add_argument("--bits", type=int, default=14)
    parser.add_argument(
        "--environment",
        default="parallel_robot",
        help="parallel_robot, zdt2, zdt6, kursawe, osyczka_kundu, or a registered custom problem.",
    )
    parser.add_argument(
        "--workspace-method",
        choices=("PRBD", "PABD"),
        default="PRBD",
        help="Workspace boundary method; PRBD is the paper/default backend.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--critic-hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--output-dir", default="runs/seed_42")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="parallel-robot-moo")
    parser.add_argument("--wandb-name", default=None)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
