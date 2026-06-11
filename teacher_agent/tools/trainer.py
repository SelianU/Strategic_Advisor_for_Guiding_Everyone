"""
teacher_agent/tools/trainer.py
────────────────────────────────
학습(train) · 평가(evaluate) · 테스트(test) 루프.

변경 사항:
  - env.step() / env.reset() 의 info["lives"] 를 추출해 agent에 전달.
  - replay_buffer.push() 에 lives, next_lives 포함.
  - select_action(obs, lives) 호출.
"""
import os
import time
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config  import Config
from .env     import make_env
from .agent   import TeacherSACAgent
from .reward  import compute_teacher_reward


# ── 평가 ──────────────────────────────────────────────────────────────────────

def evaluate(agent: TeacherSACAgent, config: Config, n_episodes: int = None):
    """reward clipping 없는 환경에서 개입 없이 실제 점수 측정."""
    n_episodes = n_episodes or config.EVAL_EPISODES

    eval_cfg = Config()
    eval_cfg.REWARD_CLIP           = False
    eval_cfg.TERMINAL_ON_LIFE_LOSS = False
    env = make_env(eval_cfg)

    agent.actor.eval()
    scores = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        lives = info.get("lives", config.MAX_LIVES)
        done  = False
        score = 0.0
        while not done:
            prob, intervene = agent.select_action(obs, lives)
            # 평가 시에는 intervention 행동을 실제 게임 action으로 사용하지 않음.
            # 여기서는 NOOP(0)으로 단순 처리 — 실제 student agent 연동 시 수정.
            obs, reward, terminated, truncated, info = env.step(0)
            lives = info.get("lives", lives)
            done  = terminated or truncated
            score += reward
        scores.append(score)

    env.close()
    agent.actor.train()
    return float(np.mean(scores)), float(np.std(scores))


# ── 학습 ──────────────────────────────────────────────────────────────────────

def train(config: Config, resume_path: str = None):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{config.ENV_NAME.replace('/', '_')}__teacher_sac__{int(time.time())}"

    print(f"Device: {device}")
    print(f"Training Teacher SAC on {config.ENV_NAME}")
    print(f"  Threshold          : {config.THRESHOLD}")
    print(f"  Policy LR          : {config.POLICY_LR}")
    print(f"  Q LR               : {config.Q_LR}")
    print(f"  Batch Size         : {config.BATCH_SIZE}")
    print(f"  Replay Capacity    : {config.REPLAY_CAPACITY:,}")
    print(f"  Learning Start     : {config.LEARNING_START:,}")
    print(f"  Autotune Entropy   : {config.AUTOTUNE}")
    print(f"  Total Frames       : {config.TOTAL_FRAMES:,}")
    print()

    writer = SummaryWriter(f"runs/{run_name}")
    env    = make_env(config)
    agent  = TeacherSACAgent(env, config, device)

    if resume_path:
        agent.load(resume_path)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    episode_rewards = deque(maxlen=100)
    qf_losses       = deque(maxlen=500)
    actor_losses    = deque(maxlen=500)
    best_eval_score = -float("inf")
    step = episode  = 0
    start_time      = time.time()

    obs, info = env.reset()
    lives     = info.get("lives", config.MAX_LIVES)
    ep_reward = 0.0

    print("Starting training...")

    while agent.total_frames < config.TOTAL_FRAMES:

        # ── 행동 선택 ────────────────────────────────────────────────────────
        if agent.total_frames < config.LEARNING_START:
            # 탐험 구간: 랜덤 intervention 확률
            action = float(np.random.uniform(0.0, 1.0))
        else:
            action, _ = agent.select_action(obs, lives)

        next_obs, game_reward, terminated, truncated, info = env.step(
            # teacher는 게임 action을 직접 내지 않음 — 여기서는 NOOP으로 수집
            # 실제 운용 시 student agent의 action을 env에 전달해야 함
            0
        )
        next_lives = info.get("lives", lives)
        done       = terminated or truncated
        intervened = (action >= config.THRESHOLD)

        # ── Teacher Reward ────────────────────────────────────────────────────────────────────────────
        teacher_reward = compute_teacher_reward(
            obs        = obs,
            lives      = lives,
            action     = action,
            intervened = intervened,
            next_obs   = next_obs,
            next_lives = next_lives,
            info       = info,
        )
        # ────────────────────────────────────────────────────────────────────────────

        agent.replay_buffer.push(obs, lives, action, teacher_reward, next_obs, next_lives, done)

        obs   = next_obs
        lives = next_lives
        agent.total_frames += config.FRAME_SKIP
        step      += 1
        ep_reward += teacher_reward

        # ── 학습 ─────────────────────────────────────────────────────────────
        if step % config.UPDATE_FREQ == 0:
            result = agent.train_step(step)
            if result is not None:
                qf_loss, actor_loss, alpha_loss = result
                qf_losses.append(qf_loss)
                actor_losses.append(actor_loss)
                if step % 100 == 0:
                    writer.add_scalar("losses/qf_loss",    qf_loss,     agent.total_frames)
                    writer.add_scalar("losses/actor_loss", actor_loss,  agent.total_frames)
                    writer.add_scalar("losses/alpha",      agent.alpha, agent.total_frames)
                    if alpha_loss is not None:
                        writer.add_scalar("losses/alpha_loss", alpha_loss, agent.total_frames)
                    fps = agent.total_frames / (time.time() - start_time + 1e-8)
                    writer.add_scalar("charts/SPS", fps, agent.total_frames)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if agent.total_frames % config.CHECKPOINT_FREQ == 0:
            path = os.path.join(
                config.CHECKPOINT_DIR,
                f"teacher_sac_frame_{agent.total_frames}.pth",
            )
            agent.save(path)

        # ── Evaluation ────────────────────────────────────────────────────────
        if (agent.total_frames % config.EVAL_FREQ == 0
                and len(agent.replay_buffer) >= config.LEARNING_START):
            eval_mean, eval_std = evaluate(agent, config)
            print(f"  >> Eval @ frame {agent.total_frames:,}: "
                  f"score = {eval_mean:.1f} +/- {eval_std:.1f}")
            writer.add_scalar("charts/eval_score", eval_mean, agent.total_frames)
            if eval_mean > best_eval_score:
                best_eval_score = eval_mean
                agent.save(
                    os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                    extra={"eval_score": eval_mean},
                )

        # ── Episode 종료 ──────────────────────────────────────────────────────
        if done:
            episode += 1
            episode_rewards.append(ep_reward)
            writer.add_scalar("charts/episodic_return", ep_reward, agent.total_frames)

            if episode % config.LOG_FREQ == 0:
                elapsed = time.time() - start_time
                fps     = agent.total_frames / (elapsed + 1e-8)
                avg_r   = np.mean(episode_rewards)
                avg_qf  = np.mean(qf_losses)    if qf_losses    else 0.0
                avg_act = np.mean(actor_losses)  if actor_losses else 0.0
                status  = (
                    "filling" if len(agent.replay_buffer) < config.LEARNING_START
                    else "training"
                )
                print(
                    f"Frame: {agent.total_frames:>10,}/{config.TOTAL_FRAMES:,} | "
                    f"Ep: {episode:>5} | "
                    f"Reward: {ep_reward:>7.1f} | "
                    f"Avg(100): {avg_r:>7.1f} | "
                    f"QF_Loss: {avg_qf:.4f} | "
                    f"Act_Loss: {avg_act:.4f} | "
                    f"Alpha: {agent.alpha:.4f} | "
                    f"Buffer: {len(agent.replay_buffer):>7,} | "
                    f"FPS: {fps:>6.1f} | "
                    f"{status}"
                )

            obs, info = env.reset()
            lives     = info.get("lives", config.MAX_LIVES)
            ep_reward = 0.0

    agent.save(os.path.join(config.CHECKPOINT_DIR, "final_model.pth"))
    env.close()
    writer.close()
    print("Training complete.")


# ── 테스트 ────────────────────────────────────────────────────────────────────

def test(model_path: str, config: Config, n_episodes: int = 10, render: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_cfg = Config()
    test_cfg.REWARD_CLIP           = False
    test_cfg.TERMINAL_ON_LIFE_LOSS = False
    env = make_env(test_cfg, render=render)

    agent = TeacherSACAgent(env, config, device)
    agent.load(model_path)
    agent.actor.eval()

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        lives     = info.get("lives", config.MAX_LIVES)
        done      = False
        score     = 0.0
        intervene_count = 0
        step_count      = 0
        while not done:
            prob, intervene = agent.select_action(obs, lives)
            obs, reward, terminated, truncated, info = env.step(0)
            lives = info.get("lives", lives)
            done  = terminated or truncated
            score += reward
            step_count += 1
            if intervene:
                intervene_count += 1
        print(
            f"  Episode {ep:>2}/{n_episodes}: "
            f"score = {score:.0f}  |  "
            f"intervene = {intervene_count}/{step_count} "
            f"({100*intervene_count/max(step_count,1):.1f}%)"
        )

    env.close()
