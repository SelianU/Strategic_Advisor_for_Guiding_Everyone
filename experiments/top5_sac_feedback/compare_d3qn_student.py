"""
Compare baseline D3QN vs. top-k teacher-feedback-guided D3QN on Space Invaders.

The experiment trains two independent D3QN+PER students:
  1. Baseline: standard D3QN updates only.
  2. Guided: standard D3QN updates plus replayed teacher distillation on states
     where the pre-trained D3QN teacher assigns the largest action gap.

The guided run stores top-k episode moments in a feedback buffer. During regular
D3QN training, it samples that buffer and adds a KL distillation objective from
teacher Q-value preferences to student Q-values.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
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

from ai_agents.d3qn_helper import load_d3qn  # noqa: E402
from experiments.d3qn.agent import D3QNAgent  # noqa: E402
from experiments.d3qn.config import Config  # noqa: E402
from experiments.d3qn.env import make_env  # noqa: E402


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


class FeedbackBuffer:
    """Circular buffer of teacher-labeled states for repeated distillation."""

    def __init__(self, capacity: int, state_shape: tuple[int, ...], n_actions: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.teacher_q = np.zeros((capacity, n_actions), dtype=np.float32)
        self.gaps = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push_batch(
        self,
        states: np.ndarray,
        teacher_q: np.ndarray,
        gaps: np.ndarray,
    ) -> None:
        for state, q_values, gap in zip(states, teacher_q, gaps):
            self.states[self.pos] = state
            self.teacher_q[self.pos] = q_values
            self.gaps[self.pos] = gap
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.states[idx], self.teacher_q[idx], self.gaps[idx]

    def __len__(self) -> int:
        return self.size


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
    cfg.TRAIN_FREQ = args.train_freq
    cfg.TARGET_UPDATE_FREQ = args.target_update_freq
    cfg.CHECKPOINT_FREQ = args.checkpoint_freq
    cfg.CHECKPOINT_DIR = str(args.output_dir / "d3qn_checkpoints")
    cfg.BEST_CHECKPOINT_DIR = str(args.output_dir / "d3qn_best_checkpoints")
    cfg.TERMINAL_ON_LIFE_LOSS = args.life_loss
    return cfg


def checkpoint_path(output_dir: Path, label: str, frame: int) -> Path:
    checkpoint_dir = output_dir / "d3qn_checkpoints" / label.lower()
    return checkpoint_dir / f"{label.lower()}_frame_{frame:012d}.pth"


def save_rolling_checkpoint(
    agent: D3QNAgent,
    output_dir: Path,
    label: str,
    keep: int,
    extra: dict | None = None,
) -> Path:
    path = checkpoint_path(output_dir, label, agent.total_frames)
    agent.save(str(path), extra={"label": label, **(extra or {})})

    checkpoints = sorted(path.parent.glob(f"{label.lower()}_frame_*.pth"))
    while len(checkpoints) > keep:
        old = checkpoints.pop(0)
        old.unlink()
        print(f"  [Removed old checkpoint] {old}")
    return path


def evaluate_d3qn(agent: D3QNAgent, cfg: Config) -> tuple[float, float]:
    eval_cfg = Config()
    eval_cfg.ENV_NAME = cfg.ENV_NAME
    eval_cfg.FRAME_SKIP = cfg.FRAME_SKIP
    eval_cfg.REWARD_CLIP = False
    eval_cfg.TERMINAL_ON_LIFE_LOSS = False
    env = make_env(eval_cfg)

    was_training = agent.q_network.training
    agent.q_network.eval()
    scores = []
    for _ in range(cfg.EVAL_EPISODES):
        state, _ = env.reset()
        done = False
        score = 0.0
        while not done:
            action = agent.select_action(state, epsilon=cfg.EVAL_EPSILON)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)

    env.close()
    if was_training:
        agent.q_network.train()
    return float(np.mean(scores)), float(np.std(scores))


def add_topk_feedback_examples(
    feedback_buffer: FeedbackBuffer,
    device: torch.device,
    teacher: torch.nn.Module,
    episode_states: list[np.ndarray],
    episode_actions: list[int],
    top_k: int,
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

    states_np = np.stack(states)
    state_tensor = torch.as_tensor(states_np, dtype=torch.uint8, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)

    with torch.no_grad():
        teacher_q = teacher(state_tensor)
        teacher_best_q, teacher_best_action = teacher_q.max(dim=1)
        student_action_q = teacher_q.gather(1, action_tensor.view(-1, 1)).squeeze(1)
        gaps = teacher_best_q - student_action_q
        k = min(top_k, gaps.numel())
        top_values, top_indices = torch.topk(gaps, k=k)

    if top_indices.numel() == 0:
        return None

    top_idx_np = top_indices.cpu().numpy()
    feedback_buffer.push_batch(
        states=states_np[top_idx_np],
        teacher_q=teacher_q[top_indices].detach().cpu().numpy(),
        gaps=top_values.detach().cpu().numpy(),
    )

    return {
        "mode": "buffered_kl",
        "added": int(top_indices.numel()),
        "buffer_size": int(len(feedback_buffer)),
        "mean_teacher_gap": float(top_values.mean().item()),
        "max_teacher_gap": float(top_values.max().item()),
        "teacher_best_action_counts": {
            str(int(action)): int(count)
            for action, count in zip(
                *np.unique(teacher_best_action[top_indices].cpu().numpy(), return_counts=True)
            )
        },
        "top_k": int(top_indices.numel()),
    }


def train_step_with_feedback(
    agent: D3QNAgent,
    feedback_buffer: FeedbackBuffer,
    feedback_batch_size: int,
    feedback_weight: float,
    distill_temperature: float,
) -> tuple[float, float | None]:
    if len(agent.replay_buffer) < agent.cfg.LEARNING_START:
        return None, None

    beta = agent.get_beta()
    states, actions, rewards, next_states, dones, leaf_idxs, is_weights = (
        agent.replay_buffer.sample(agent.cfg.BATCH_SIZE, beta)
    )

    s = torch.as_tensor(states, dtype=torch.uint8, device=agent.device)
    ns = torch.as_tensor(next_states, dtype=torch.uint8, device=agent.device)
    a = torch.as_tensor(actions, dtype=torch.long, device=agent.device).unsqueeze(1)
    r = torch.as_tensor(rewards, dtype=torch.float32, device=agent.device)
    d = torch.as_tensor(dones, dtype=torch.float32, device=agent.device)
    w = torch.as_tensor(is_weights, dtype=torch.float32, device=agent.device)

    current_q = agent.q_network(s).gather(1, a).squeeze(1)

    with torch.no_grad():
        next_a = agent.q_network(ns).argmax(1, keepdim=True)
        next_q = agent.target_network(ns).gather(1, next_a).squeeze(1)
        target_q = r + agent.cfg.DISCOUNT_FACTOR * next_q * (1.0 - d)

    td_errors = (target_q - current_q).detach().cpu().numpy()
    elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
    td_loss = (w * elementwise_loss).mean()

    distill_loss = None
    if len(feedback_buffer) >= feedback_batch_size:
        fb_states, fb_teacher_q, _ = feedback_buffer.sample(feedback_batch_size)
        fb_s = torch.as_tensor(fb_states, dtype=torch.uint8, device=agent.device)
        fb_teacher_q_t = torch.as_tensor(
            fb_teacher_q,
            dtype=torch.float32,
            device=agent.device,
        )
        temp = distill_temperature
        student_log_probs = F.log_softmax(agent.q_network(fb_s) / temp, dim=1)
        teacher_probs = F.softmax(fb_teacher_q_t / temp, dim=1)
        distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        distill_loss = distill_loss * (temp * temp)
        loss = td_loss + feedback_weight * distill_loss
    else:
        loss = td_loss

    agent.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), agent.cfg.GRAD_CLIP_NORM)
    agent.optimizer.step()
    agent.replay_buffer.update_priorities(leaf_idxs, td_errors, agent.cfg.PER_EPS)

    return float(td_loss.item()), float(distill_loss.item()) if distill_loss is not None else None


def train_one_student(
    label: str,
    cfg: Config,
    args: argparse.Namespace,
    device: torch.device,
    teacher: torch.nn.Module | None,
    seed: int,
    completed_results: list[RunResult],
    resume_path: Path | None = None,
) -> RunResult:
    set_seed(seed)
    env = make_env(cfg)
    agent = D3QNAgent(env, cfg, device)
    if resume_path is not None:
        agent.load(str(resume_path))

    eval_history: list[dict] = []
    episode_history: list[dict] = []
    losses = deque(maxlen=500)
    feedback_losses = deque(maxlen=500)
    feedback_buffer = None
    if teacher is not None:
        feedback_buffer = FeedbackBuffer(
            capacity=args.feedback_buffer_size,
            state_shape=env.observation_space.shape,
            n_actions=env.action_space.n,
        )

    state, _ = env.reset()
    episode_reward = 0.0
    episode_states: list[np.ndarray] = []
    episode_actions: list[int] = []
    episode = 0
    step = 0
    next_eval_frame = (agent.total_frames // cfg.EVAL_FREQ + 1) * cfg.EVAL_FREQ
    next_checkpoint_frame = (
        (agent.total_frames // cfg.CHECKPOINT_FREQ + 1) * cfg.CHECKPOINT_FREQ
        if cfg.CHECKPOINT_FREQ > 0
        else cfg.TOTAL_FRAMES
    )
    started_at = time.time()

    print(f"\n[{label}] D3QN training start")
    while agent.total_frames < cfg.TOTAL_FRAMES:
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

        if step % cfg.TRAIN_FREQ == 0:
            if feedback_buffer is not None:
                loss, fb_loss = train_step_with_feedback(
                    agent=agent,
                    feedback_buffer=feedback_buffer,
                    feedback_batch_size=args.feedback_batch_size,
                    feedback_weight=args.feedback_weight,
                    distill_temperature=args.distill_temperature,
                )
                if loss is not None:
                    losses.append(loss)
                if fb_loss is not None:
                    feedback_losses.append(fb_loss)
            else:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

        if cfg.SOFT_UPDATE:
            agent.update_target(soft=True, tau=cfg.TAU)
        elif step % cfg.TARGET_UPDATE_FREQ == 0:
            agent.update_target(soft=False)
            print(f"[{label}] target updated @ frame={agent.total_frames:,}")

        while agent.total_frames >= next_eval_frame:
            if len(agent.replay_buffer) >= cfg.LEARNING_START:
                mean_score, std_score = evaluate_d3qn(agent, cfg)
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
                current_result = RunResult(
                    label=label,
                    eval_history=eval_history,
                    episode_history=episode_history,
                    final_frame=agent.total_frames,
                )
                write_histories([*completed_results, current_result], args.output_dir)
                if not args.no_plot:
                    plot_path = plot_results([*completed_results, current_result], args.output_dir)
                    print(f"[{label}] updated plot: {plot_path}")
            next_eval_frame += cfg.EVAL_FREQ

        if cfg.CHECKPOINT_FREQ > 0 and agent.total_frames >= next_checkpoint_frame:
            save_rolling_checkpoint(
                agent=agent,
                output_dir=args.output_dir,
                label=label,
                keep=args.keep_checkpoints,
                extra={"kind": "rolling"},
            )
            while next_checkpoint_frame <= agent.total_frames:
                next_checkpoint_frame += cfg.CHECKPOINT_FREQ

        if done:
            episode += 1
            feedback = None
            if (
                teacher is not None
                and len(agent.replay_buffer) >= cfg.LEARNING_START
                and agent.total_frames >= args.feedback_start
            ):
                feedback = add_topk_feedback_examples(
                    feedback_buffer=feedback_buffer,
                    device=device,
                    teacher=teacher,
                    episode_states=episode_states,
                    episode_actions=episode_actions,
                    top_k=args.top_k,
                    max_episode_states=args.max_episode_states,
                )

            episode_history.append({
                "label": label,
                "episode": episode,
                "frame": int(agent.total_frames),
                "episode_reward": float(episode_reward),
                "loss": float(np.mean(losses)) if losses else None,
                "feedback_loss": float(np.mean(feedback_losses)) if feedback_losses else None,
                "epsilon": float(agent.get_epsilon()),
                "beta": float(agent.get_beta()),
                "feedback": feedback,
            })

            if episode % args.log_episodes == 0:
                fps = agent.total_frames / (time.time() - started_at + 1e-8)
                avg_loss = np.mean(losses) if losses else 0.0
                suffix = ""
                if feedback:
                    suffix = (
                        f" | fb_kl={np.mean(feedback_losses):.4f}"
                        if feedback_losses else " | fb_kl=n/a"
                    )
                    suffix += (
                        f" gap={feedback['mean_teacher_gap']:.3f}"
                        f" fb_buf={feedback['buffer_size']}"
                    )
                print(
                    f"[{label}] frame={agent.total_frames:,}/{cfg.TOTAL_FRAMES:,} "
                    f"ep={episode} reward={episode_reward:.1f} "
                    f"loss={avg_loss:.4f} eps={agent.get_epsilon():.4f} "
                    f"fps={fps:.1f}{suffix}"
                )

            state, _ = env.reset()
            episode_reward = 0.0
            episode_states = []
            episode_actions = []

    save_rolling_checkpoint(
        agent=agent,
        output_dir=args.output_dir,
        label=label,
        keep=args.keep_checkpoints,
        extra={"kind": "final"},
    )

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
    (output_dir / "d3qn_eval_history.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "d3qn_eval_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "frame", "mean_reward", "std_reward"],
        )
        writer.writeheader()
        for result in results:
            writer.writerows(result.eval_history)


def plot_results(results: list[RunResult], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    has_data = False
    for result in results:
        frames = [row["frame"] for row in result.eval_history]
        rewards = [row["mean_reward"] for row in result.eval_history]
        stds = [row["std_reward"] for row in result.eval_history]
        if not frames:
            continue
        has_data = True
        ax.plot(frames, rewards, marker="o", label=result.label)
        lo = np.array(rewards) - np.array(stds)
        hi = np.array(rewards) + np.array(stds)
        ax.fill_between(frames, lo, hi, alpha=0.15)

    ax.set_title("D3QN Student: Baseline vs Buffered Teacher KL Feedback")
    ax.set_xlabel("Total Frames")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.grid(True, alpha=0.3)
    if has_data:
        ax.legend()
    fig.tight_layout()

    plot_path = output_dir / "d3qn_student_comparison.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline D3QN and buffered teacher-feedback-guided D3QN students.",
    )
    parser.add_argument("--total-frames", type=int, default=1_000_000)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=15,
        help="Action repeat. Default 15 matches the teacher training setup.",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--learning-start", type=int, default=50_000)
    parser.add_argument("--feedback-start", type=int, default=50_000)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-freq", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--feedback-weight", type=float, default=0.3)
    parser.add_argument("--feedback-buffer-size", type=int, default=50_000)
    parser.add_argument("--feedback-batch-size", type=int, default=16)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--max-episode-states", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-episodes", type=int, default=10)
    parser.add_argument("--teacher-path", type=Path, default=TEACHER_PATH)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5_000_000,
        help="Save a rolling checkpoint every N frames. Use 0 to disable periodic saves.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="Number of latest checkpoints to keep per run label.",
    )
    parser.add_argument(
        "--resume-guided",
        type=Path,
        default=None,
        help="Resume Guided from a D3QN checkpoint.",
    )
    parser.add_argument(
        "--resume-baseline",
        type=Path,
        default=None,
        help="Resume Baseline from a D3QN checkpoint.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Deprecated: rolling checkpoints are always saved at final frame.",
    )
    parser.add_argument(
        "--life-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate training episodes on life loss. Default true matches the teacher.",
    )
    parser.add_argument("--guided-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.guided_only and args.baseline_only:
        raise ValueError("--guided-only and --baseline-only are mutually exclusive.")
    if args.checkpoint_freq < 0:
        raise ValueError("--checkpoint-freq must be >= 0.")
    if args.keep_checkpoints < 1:
        raise ValueError("--keep-checkpoints must be >= 1.")
    if args.resume_guided is not None and not args.resume_guided.exists():
        raise FileNotFoundError(f"Guided checkpoint not found: {args.resume_guided}")
    if args.resume_baseline is not None and not args.resume_baseline.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.resume_baseline}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_config(args)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Device       : {device}")
    print(f"Total frames : {cfg.TOTAL_FRAMES:,}")
    print(f"Frame skip   : {cfg.FRAME_SKIP}")
    print(f"Life loss    : {cfg.TERMINAL_ON_LIFE_LOSS}")
    print(f"Eval freq    : {cfg.EVAL_FREQ:,}")
    print(f"Feedback w   : {args.feedback_weight}")
    print(f"Top k        : {args.top_k}")
    print(f"Feedback buf : {args.feedback_buffer_size:,}")
    print(f"Feedback bs  : {args.feedback_batch_size}")
    print(f"Distill temp : {args.distill_temperature}")
    print(f"Checkpoint f : {args.checkpoint_freq:,}")
    print(f"Keep ckpts   : {args.keep_checkpoints}")

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
    if not args.baseline_only:
        results.append(
            train_one_student(
                label="Guided",
                cfg=cfg,
                args=args,
                device=device,
                teacher=teacher,
                seed=args.seed,
                completed_results=results,
                resume_path=args.resume_guided,
            )
        )

    if not args.guided_only:
        results.append(
            train_one_student(
                label="Baseline",
                cfg=cfg,
                args=args,
                device=device,
                teacher=None,
                seed=args.seed,
                completed_results=results,
                resume_path=args.resume_baseline,
            )
        )

    write_histories(results, args.output_dir)
    print(f"\nSaved histories: {args.output_dir / 'd3qn_eval_history.json'}")
    print(f"Saved CSV      : {args.output_dir / 'd3qn_eval_history.csv'}")

    if not args.no_plot:
        plot_path = plot_results(results, args.output_dir)
        print(f"Saved plot     : {plot_path}")


if __name__ == "__main__":
    main()
