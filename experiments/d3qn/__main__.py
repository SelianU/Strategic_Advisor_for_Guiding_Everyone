"""
D3QN (Dueling Double DQN + Prioritized Experience Replay)
for ALE Atari games

Base papers:
  [1] Mnih et al.       (2013) - DQN
  [2] van Hasselt et al.(2015) - Double DQN
  [3] Wang et al.       (2016) - Dueling Network
  [4] Schaul et al.     (2015) - Prioritized Experience Replay

Usage:
    python -m d3qn --train --game SpaceInvaders
    python -m d3qn --train --game Breakout --resume checkpoints_d3qn_Breakout/best_model.pth
    python -m d3qn --test  --game SpaceInvaders --model checkpoints_d3qn_SpaceInvaders/best_model.pth
    python -m d3qn --test  --game SpaceInvaders --model checkpoints_d3qn_SpaceInvaders/best_model.pth --render
"""

import sys
import argparse

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
torch.set_float32_matmul_precision('high')

from .config import Config
from .env import make_env
from .agent import D3QNAgent
from .runner import train


def test(model_path: str, config: Config, n_episodes: int = 10,
         render: bool = False, render_scale: float = 2.0, fps: int = 60,
         record_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_cfg = Config()
    test_cfg.ENV_NAME              = config.ENV_NAME   # 지정된 게임 유지
    test_cfg.REWARD_CLIP           = False
    test_cfg.TERMINAL_ON_LIFE_LOSS = False

    # ── 1단계: render 없이 n_episodes 측정 ──────────────────
    env   = make_env(test_cfg)
    agent = D3QNAgent(env, config, device)
    agent.load(model_path)
    agent.q_network.eval()

    scores = []
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        done  = False
        score = 0.0
        while not done:
            action = agent.select_action(state, epsilon=config.EVAL_EPSILON)
            state, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            score += reward
        scores.append(score)
        print(f'  Episode {ep:>2}/{n_episodes}: score = {score:.0f}')

    env.close()
    print(f'\nResults over {n_episodes} episodes:')
    print(f'  Mean   : {np.mean(scores):.1f}')
    print(f'  Std    : {np.std(scores):.1f}')
    print(f'  Min    : {np.min(scores):.0f}')
    print(f'  Max    : {np.max(scores):.0f}')
    print(f'  Median : {np.median(scores):.0f}')

    # ── 2단계: render / record 모드로 1 에피소드 시연 ─────
    if not render and not record_path:
        return

    if render:
        print('\n--- 렌더링 시연 시작 (창 닫으면 종료) ---')
    else:
        print(f'\n--- 녹화 중 ({record_path}) ---')
    render_env = make_env(test_cfg, render=render, render_scale=render_scale, fps=fps,
                          record_path=record_path)
    state, _   = render_env.reset()
    done  = False
    score = 0.0
    while not done:
        action = agent.select_action(state, epsilon=config.EVAL_EPSILON)
        state, reward, terminated, truncated, _ = render_env.step(action)
        done   = terminated or truncated
        score += reward
    render_env.close()
    print(f'시연 에피소드 점수: {score:.0f}')


def main():
    parser = argparse.ArgumentParser(
        description='D3QN — Dueling Double DQN + PER, FRAME_SKIP=15',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--train',        action='store_true')
    parser.add_argument('--test',         action='store_true')
    parser.add_argument('--game',           type=str,   default=None,
                        help='ALE 게임 이름 (예: SpaceInvaders, Breakout, Pong). '
                             '지정 시 checkpoint 폴더는 checkpoints_d3qn_{game}')
    parser.add_argument('--checkpoint-dir',      type=str, default=None,
                        help='rolling 체크포인트 폴더 직접 지정 (--game보다 우선)')
    parser.add_argument('--best-checkpoint-dir', type=str, default=None,
                        help='best/milestone/final 저장 폴더 직접 지정 (--game보다 우선)')
    parser.add_argument('--model',        type=str,   default=None)
    parser.add_argument('--resume',       type=str,   default=None)
    parser.add_argument('--render',       action='store_true')
    parser.add_argument('--episodes',     type=int,   default=10)
    parser.add_argument('--lr',           type=float, default=None)
    parser.add_argument('--total-frames', type=int,   default=None)
    parser.add_argument('--alpha',        type=float, default=None,  help='PER alpha (default: 0.6)')
    parser.add_argument('--beta-start',   type=float, default=None,  help='PER beta start (default: 0.4)')
    parser.add_argument('--render-scale', type=float, default=2.0,   help='렌더 창 배율 (default: 2.0 → 320×420)')
    parser.add_argument('--fps',          type=int,   default=60,    help='게임 기준 FPS (default: 60)')
    parser.add_argument('--record',       type=str,   default=None,  metavar='PATH', help='영상 저장 경로 (예: output.mp4)')

    args   = parser.parse_args()
    config = Config()

    # 게임 설정
    if args.game:
        config.ENV_NAME            = f'ALE/{args.game}-v5'
        config.CHECKPOINT_DIR      = f'./checkpoints_d3qn_{args.game}'
        config.BEST_CHECKPOINT_DIR = f'./best_model_d3qn_{args.game}'
    if args.checkpoint_dir:
        config.CHECKPOINT_DIR      = args.checkpoint_dir
    if args.best_checkpoint_dir:
        config.BEST_CHECKPOINT_DIR = args.best_checkpoint_dir

    if args.lr:           config.LEARNING_RATE = args.lr
    if args.total_frames: config.TOTAL_FRAMES  = args.total_frames
    if args.alpha:        config.PER_ALPHA      = args.alpha
    if args.beta_start:   config.PER_BETA_START = args.beta_start

    if args.train:
        train(config, resume_path=args.resume)
    elif args.test:
        if not args.model:
            parser.error('--test requires --model MODEL_PATH')
        test(args.model, config, n_episodes=args.episodes, render=args.render,
             render_scale=args.render_scale, fps=args.fps, record_path=args.record)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
