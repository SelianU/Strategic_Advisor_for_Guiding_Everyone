"""
Compare baseline SAC vs. top-5 teacher-feedback-guided SAC on Space Invaders.

The experiment trains two independent Discrete SAC students:
  1. Baseline: standard SAC updates only.
  2. Guided: standard SAC updates plus one end-of-episode actor correction on
     the five states where the D3QN teacher assigns the largest action gap.

Outputs are written under experiments/top5_sac_feedback/outputs by default:
  - eval_history.json
  - eval_history.csv
  - sac_student_comparison.png
  - optional final model checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _NullSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


if "torch.utils.tensorboard" not in sys.modules:
    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
    tensorboard_stub.SummaryWriter = _NullSummaryWriter
    sys.modules["torch.utils.tensorboard"] = tensorboard_stub

from ai_agents.d3qn_helper import load_d3qn  # noqa: E402
from training.space_invaders.sac_atari_v2 import Config, SACAgent, make_env  # noqa: E402


TEACHER_PATH = (
    PROJECT_ROOT
    / "data"
    / "checkpoints"
    / "space_invaders"
    / "best_model_spaceinvaders.pth"
)


@dataclass
class RunResult:
    label: str
    eval_history: list[dict]
    episode_history: list[dict]
    final_frame: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.TOTAL_FRAMES = args.total_frames
    cfg.FRAME_SKIP = args.frame_skip
    cfg.EVAL_FREQ = args.eval_freq
    cfg.EVAL_EPISODES = args.eval_episodes
    cfg.REPLAY_CAPACITY = args.replay_capacity
    cfg.LEARNING_START = args.learning_start
    cfg.UPDATE_FREQ = args.update_freq
    cfg.TARGET_UPDATE_FREQ = args.target_update_freq
    cfg.CHECKPOINT_FREQ = max(args.total_frames + cfg.FRAME_SKIP, 1)
    cfg.CHECKPOINT_DIR = str(args.output_dir / "checkpoints")
    cfg.TERMINAL_ON_LIFE_LOSS = args.life_loss
    return cfg


def select_eval_action(agent: SACAgent, state: np.ndarray) -> int:
    s = torch.as_tensor(state, dtype=torch.uint8, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        logits = agent.actor(s)
    return int(torch.argmax(logits, dim=1).item())


def evaluate_greedy(agent: SACAgent, n_episodes: int) -> tuple[float, float]:
    eval_cfg = Config()
    eval_cfg.FRAME_SKIP = agent.cfg.FRAME_SKIP
    eval_cfg.REWARD_CLIP = False
    eval_cfg.TERMINAL_ON_LIFE_LOSS = False
    env = make_env(eval_cfg)

    agent.actor.eval()
    scores = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        score = 0.0
        while not done:
            action = select_eval_action(agent, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)

    env.close()
    agent.actor.train()
    return float(np.mean(scores)), float(np.std(scores))


def apply_top5_feedback(
    agent: SACAgent,
    teacher: torch.nn.Module,
    episode_states: list[np.ndarray],
    episode_actions: list[int],
    top_k: int,
    feedback_weight: float,
    max_episode_states: int,
) -> dict | None:
    if not episode_states:
        return None

    states = episode_states
    actions = episode_actions
    if len(states) > max_episode_states:
        idx = np.linspace(0, len(states) - 1, num=max_episode_states, dtype=np.int64)
        states = [states[i] for i in idx]
        actions = [actions[i] for i in idx]

    state_tensor = torch.as_tensor(np.stack(states), dtype=torch.uint8, device=agent.device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=agent.device)

    with torch.no_grad():
        teacher_q = teacher(state_tensor)
        teacher_best_q, teacher_best_action = teacher_q.max(dim=1)
        student_action_q = teacher_q.gather(1, action_tensor.view(-1, 1)).squeeze(1)
        gaps = teacher_best_q - student_action_q
        k = min(top_k, gaps.numel())
        top_values, top_indices = torch.topk(gaps, k=k)

    if top_indices.numel() == 0:
        return None

    logits = agent.actor(state_tensor[top_indices])
    ce_loss = F.cross_entropy(logits, teacher_best_action[top_indices])
    loss = feedback_weight * ce_loss

    agent.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    agent.actor_optimizer.step()

    return {
        "feedback_loss": float(ce_loss.item()),
        "mean_teacher_gap": float(top_values.mean().item()),
        "max_teacher_gap": float(top_values.max().item()),
        "top_k": int(top_indices.numel()),
    }


def train_one_student(
    label: str,
    cfg: Config,
    args: argparse.Namespace,
    device: torch.device,
    teacher: torch.nn.Module | None,
    seed: int,
) -> RunResult:
    set_seed(seed)
    env = make_env(cfg)
    agent = SACAgent(env, cfg, device)

    eval_history: list[dict] = []
    episode_history: list[dict] = []
    qf_losses = deque(maxlen=500)
    actor_losses = deque(maxlen=500)

    state, _ = env.reset()
    episode_reward = 0.0
    episode_states: list[np.ndarray] = []
    episode_actions: list[int] = []
    episode = 0
    step = 0
    next_eval_frame = cfg.EVAL_FREQ
    started_at = time.time()

    print(f"\n[{label}] training start")
    while agent.total_frames < cfg.TOTAL_FRAMES:
        if agent.total_frames < cfg.LEARNING_START:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_states.append(np.array(state, copy=True))
        episode_actions.append(int(action))

        state = next_state
        episode_reward += reward
        agent.total_frames += cfg.FRAME_SKIP
        step += 1

        if step % cfg.UPDATE_FREQ == 0:
            result = agent.train_step(step)
            if result is not None:
                qf_loss, actor_loss, _ = result
                qf_losses.append(qf_loss)
                actor_losses.append(actor_loss)

        while agent.total_frames >= next_eval_frame:
            if len(agent.replay_buffer) >= cfg.LEARNING_START:
                mean_score, std_score = evaluate_greedy(agent, cfg.EVAL_EPISODES)
                eval_history.append({
                    "label": label,
                    "frame": int(next_eval_frame),
                    "mean_reward": mean_score,
                    "std_reward": std_score,
                })
                print(
                    f"[{label}] eval @ {next_eval_frame:,}: "
                    f"{mean_score:.1f} +/- {std_score:.1f}"
                )
            next_eval_frame += cfg.EVAL_FREQ

        if done:
            episode += 1
            feedback = None
            if (
                teacher is not None
                and len(agent.replay_buffer) >= cfg.LEARNING_START
                and agent.total_frames >= args.feedback_start
            ):
                feedback = apply_top5_feedback(
                    agent=agent,
                    teacher=teacher,
                    episode_states=episode_states,
                    episode_actions=episode_actions,
                    top_k=args.top_k,
                    feedback_weight=args.feedback_weight,
                    max_episode_states=args.max_episode_states,
                )

            episode_history.append({
                "label": label,
                "episode": episode,
                "frame": int(agent.total_frames),
                "episode_reward": float(episode_reward),
                "qf_loss": float(np.mean(qf_losses)) if qf_losses else None,
                "actor_loss": float(np.mean(actor_losses)) if actor_losses else None,
                "feedback": feedback,
            })

            if episode % args.log_episodes == 0:
                fps = agent.total_frames / (time.time() - started_at + 1e-8)
                avg_q = np.mean(qf_losses) if qf_losses else 0.0
                avg_actor = np.mean(actor_losses) if actor_losses else 0.0
                suffix = ""
                if feedback:
                    suffix = (
                        f" | fb_ce={feedback['feedback_loss']:.4f}"
                        f" gap={feedback['mean_teacher_gap']:.3f}"
                    )
                print(
                    f"[{label}] frame={agent.total_frames:,}/{cfg.TOTAL_FRAMES:,} "
                    f"ep={episode} reward={episode_reward:.1f} "
                    f"q={avg_q:.4f} actor={avg_actor:.4f} fps={fps:.1f}{suffix}"
                )

            state, _ = env.reset()
            episode_reward = 0.0
            episode_states = []
            episode_actions = []

    if args.save_models:
        model_path = args.output_dir / f"{label.lower()}_final_model.pth"
        agent.save(str(model_path), extra={"label": label})

    env.close()
    return RunResult(label, eval_history, episode_history, agent.total_frames)


def write_histories(results: list[RunResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [
            {
                "label": result.label,
                "final_frame": result.final_frame,
                "eval_history": result.eval_history,
                "episode_history": result.episode_history,
            }
            for result in results
        ]
    }
    (output_dir / "eval_history.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "eval_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "frame", "mean_reward", "std_reward"],
        )
        writer.writeheader()
        for result in results:
            writer.writerows(result.eval_history)


def plot_results(results: list[RunResult], output_dir: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to save the comparison plot. "
            "Install it in the active environment or use --no-plot."
        ) from exc

    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        frames = [row["frame"] for row in result.eval_history]
        rewards = [row["mean_reward"] for row in result.eval_history]
        stds = [row["std_reward"] for row in result.eval_history]
        if not frames:
            continue
        ax.plot(frames, rewards, marker="o", label=result.label)
        lo = np.array(rewards) - np.array(stds)
        hi = np.array(rewards) + np.array(stds)
        ax.fill_between(frames, lo, hi, alpha=0.15)

    ax.set_title("SAC Student: Baseline vs Top-5 Teacher Feedback")
    ax.set_xlabel("Total Frames")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / "sac_student_comparison.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline SAC and top-5 feedback-guided SAC students.",
    )
    parser.add_argument("--total-frames", type=int, default=200_000)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=15,
        help="Action repeat. Default 15 matches the D3QN teacher training setup.",
    )
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--learning-start", type=int, default=20_000)
    parser.add_argument("--feedback-start", type=int, default=20_000)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--update-freq", type=int, default=4)
    parser.add_argument("--target-update-freq", type=int, default=8_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--feedback-weight", type=float, default=1.0)
    parser.add_argument("--max-episode-states", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-episodes", type=int, default=10)
    parser.add_argument(
        "--teacher-path",
        type=Path,
        default=TEACHER_PATH,
        help="Path to the pre-trained D3QN teacher checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR / "outputs",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--life-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate training episodes on life loss. Default true matches the teacher.",
    )
    parser.add_argument(
        "--guided-only",
        action="store_true",
        help="Run only the feedback-guided student for quick smoke tests.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline student for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.guided_only and args.baseline_only:
        raise ValueError("--guided-only and --baseline-only are mutually exclusive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_config(args)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Device       : {device}")
    print(f"Total frames : {cfg.TOTAL_FRAMES:,}")
    print(f"Frame skip   : {cfg.FRAME_SKIP}")
    print(f"Life loss    : {cfg.TERMINAL_ON_LIFE_LOSS}")
    print(f"Eval freq    : {cfg.EVAL_FREQ:,}")

    teacher = None
    if not args.baseline_only:
        if not args.teacher_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_path}")
        teacher, teacher_actions = load_d3qn(
            "space_invaders",
            str(args.teacher_path),
            device=str(device),
        )
        teacher.eval()
        if teacher_actions != 6:
            raise ValueError(f"Expected Space Invaders teacher with 6 actions, got {teacher_actions}")

    results: list[RunResult] = []
    if not args.guided_only:
        results.append(
            train_one_student(
                label="Baseline",
                cfg=cfg,
                args=args,
                device=device,
                teacher=None,
                seed=args.seed,
            )
        )

    if not args.baseline_only:
        results.append(
            train_one_student(
                label="Guided",
                cfg=cfg,
                args=args,
                device=device,
                teacher=teacher,
                seed=args.seed,
            )
        )

    write_histories(results, args.output_dir)
    print(f"\nSaved histories: {args.output_dir / 'eval_history.json'}")
    print(f"Saved CSV      : {args.output_dir / 'eval_history.csv'}")

    if not args.no_plot:
        plot_path = plot_results(results, args.output_dir)
        print(f"Saved plot     : {plot_path}")


if __name__ == "__main__":
    main()
