import os
import re
import glob
import time
from collections import deque

import numpy as np
import torch

from .config import Config
from .env import make_env
from .agent import D3QNAgent


def _cleanup_checkpoints(config: Config):
    """CHECKPOINT_DIR 내 d3qn_frame_*.pth가 MAX_CHECKPOINTS를 초과하면 가장 오래된 것 삭제."""
    pattern = os.path.join(config.CHECKPOINT_DIR, 'd3qn_frame_*.pth')
    files   = glob.glob(pattern)

    def frame_num(path):
        m = re.search(r'd3qn_frame_(\d+)\.pth$', path)
        return int(m.group(1)) if m else 0

    files.sort(key=frame_num)
    while len(files) > config.MAX_CHECKPOINTS:
        oldest = files.pop(0)
        os.remove(oldest)
        print(f'  [Cleanup] {os.path.basename(oldest)} 삭제')


def evaluate(agent: D3QNAgent, config: Config, n_episodes: int = None) -> tuple:
    """Reward clipping 없는 별도 환경에서 실제 게임 점수 수집."""
    n_episodes = n_episodes or config.EVAL_EPISODES

    eval_cfg = Config()
    eval_cfg.ENV_NAME              = config.ENV_NAME
    eval_cfg.REWARD_CLIP           = False
    eval_cfg.TERMINAL_ON_LIFE_LOSS = False
    env = make_env(eval_cfg)

    scores = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done  = False
        score = 0.0
        while not done:
            action = agent.select_action(state, epsilon=config.EVAL_EPSILON)
            state, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            score += reward
        scores.append(score)

    env.close()
    return float(np.mean(scores)), float(np.std(scores))


def train(config: Config, resume_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Training D3QN on {config.ENV_NAME}')
    print(f'  Learning Rate      : {config.LEARNING_RATE}')
    print(f'  Optimizer          : Adam (eps={config.ADAM_EPS})')
    print(f'  Train Freq         : every {config.TRAIN_FREQ} steps')
    print(f'  Replay Buffer      : {config.REPLAY_CAPACITY:,}')
    print(f'  PER alpha          : {config.PER_ALPHA}')
    print(f'  PER beta           : {config.PER_BETA_START} -> 1.0 over {config.PER_BETA_FRAMES:,} frames')
    print(f'  Learning Start     : {config.LEARNING_START:,} transitions')
    print(f'  Target Update      : Hard (every {config.TARGET_UPDATE_FREQ} steps)')
    print(f'  Epsilon            : {config.EPSILON_START} -> {config.EPSILON_END} over {config.EPSILON_DECAY:,} frames')
    print(f'  Frame Skip         : {config.FRAME_SKIP}')
    print(f'  Reward Clip        : {config.REWARD_CLIP}')
    print(f'  TerminalOnLifeLoss : {config.TERMINAL_ON_LIFE_LOSS}')
    print(f'  Total Frames       : {config.TOTAL_FRAMES:,}')
    print(f'  Checkpoint Dir     : {config.CHECKPOINT_DIR}  (최근 {config.MAX_CHECKPOINTS}개 rolling)')
    print(f'  Best Checkpoint Dir: {config.BEST_CHECKPOINT_DIR}  (best / milestone / final)')
    print()

    env   = make_env(config)
    agent = D3QNAgent(env, config, device)

    if resume_path:
        agent.load(resume_path)

    os.makedirs(config.CHECKPOINT_DIR,      exist_ok=True)
    os.makedirs(config.BEST_CHECKPOINT_DIR, exist_ok=True)

    episode_rewards = deque(maxlen=100)
    losses          = deque(maxlen=500)
    best_eval_score = -float('inf')
    step = episode  = 0
    start_time      = time.time()
    start_frames    = agent.total_frames  # FPS 계산용

    _f = agent.total_frames
    next_checkpoint_frame = (_f // config.CHECKPOINT_FREQ + 1) * config.CHECKPOINT_FREQ
    next_eval_frame       = (_f // config.EVAL_FREQ       + 1) * config.EVAL_FREQ
    next_milestone_frame  = (_f // config.MILESTONE_FREQ  + 1) * config.MILESTONE_FREQ

    state, _  = env.reset()
    ep_reward     = 0.0
    ep_reward_raw = 0.0

    print('Starting training...')

    while agent.total_frames < config.TOTAL_FRAMES:

        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        agent.total_frames += config.FRAME_SKIP
        step          += 1
        ep_reward     += reward
        ep_reward_raw += info.get('raw_reward', reward)

        if step % config.TRAIN_FREQ == 0:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        if config.SOFT_UPDATE:
            agent.update_target(soft=True, tau=config.TAU)
        elif step % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target(soft=False)
            print(f'  [Target Hard Updated] frame={agent.total_frames:,}')

        # 일반 체크포인트 (250K 간격, 최대 10개 롤링) → CHECKPOINT_DIR
        # 동시에 BEST_CHECKPOINT_DIR에도 최신 1개 유지
        if agent.total_frames >= next_checkpoint_frame:
            fname = f'd3qn_frame_{agent.total_frames}.pth'
            agent.save(os.path.join(config.CHECKPOINT_DIR, fname))
            _cleanup_checkpoints(config)
            # best dir에 최신 1개만 유지
            for old in glob.glob(os.path.join(config.BEST_CHECKPOINT_DIR, 'd3qn_frame_*.pth')):
                os.remove(old)
            agent.save(os.path.join(config.BEST_CHECKPOINT_DIR, fname))
            next_checkpoint_frame += config.CHECKPOINT_FREQ

        # 마일스톤 체크포인트 (50M 간격, 영구 보관) → BEST_CHECKPOINT_DIR
        if agent.total_frames >= next_milestone_frame:
            path = os.path.join(config.BEST_CHECKPOINT_DIR, f'd3qn_milestone_{agent.total_frames}.pth')
            agent.save(path)
            print(f'  [Milestone] {os.path.basename(path)} 저장')
            next_milestone_frame += config.MILESTONE_FREQ

        if agent.total_frames >= next_eval_frame:
            next_eval_frame += config.EVAL_FREQ
            if len(agent.replay_buffer) >= config.LEARNING_START:
                eval_mean, eval_std = evaluate(agent, config)
                print(f'  >> Eval @ frame {agent.total_frames:,}: '
                      f'score = {eval_mean:.1f} +/- {eval_std:.1f}')
                if eval_mean > best_eval_score:
                    best_eval_score = eval_mean
                    best_path = os.path.join(config.BEST_CHECKPOINT_DIR, 'best_model.pth')
                    agent.save(best_path, extra={'eval_score': eval_mean})

        if done:
            episode += 1
            episode_rewards.append(ep_reward)

            if episode % config.LOG_FREQ == 0:
                elapsed      = time.time() - start_time
                new_frames   = agent.total_frames - start_frames
                fps          = new_frames / elapsed if elapsed > 0 else 0
                avg_r        = np.mean(episode_rewards)
                avg_l        = np.mean(losses) if losses else 0.0
                buf_pct      = len(agent.replay_buffer) / config.REPLAY_CAPACITY * 100
                status       = 'filling' if len(agent.replay_buffer) < config.LEARNING_START \
                               else 'training'
                eps          = agent.get_epsilon()
                eps_str      = f'eps: {eps:.4f} | ' if eps > config.EPSILON_END else ''
                print(
                    f'Frame: {agent.total_frames:>10,}/{config.TOTAL_FRAMES:,} | '
                    f'Ep: {episode:>5} | '
                    f'Reward: {ep_reward:>6.1f} ({ep_reward_raw:>6.1f}) | '
                    f'Avg(100): {avg_r:>6.1f} | '
                    f'Loss: {avg_l:.4f} | '
                    + eps_str +
                    f'beta: {agent.get_beta():.4f} | '
                    f'Buffer: {buf_pct:>5.1f}% | '
                    f'FPS: {fps:>6.1f} | '
                    f'{status}'
                )

            state, _      = env.reset()
            ep_reward     = 0.0
            ep_reward_raw = 0.0

    agent.save(os.path.join(config.BEST_CHECKPOINT_DIR, 'final_model.pth'))
    env.close()
    print('Training complete.')
