import os
import sys
import random
import argparse
import time
from collections import deque

import numpy as np
import cv2

import gymnasium as gym
import ale_py  # noqa: F401  ← ALE 네임스페이스 등록 (NamespaceNotFound 해결)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.stdout.reconfigure(encoding='utf-8')

# TF32 가속 (Ampere+ GPU: RTX 30xx/40xx)
torch.set_float32_matmul_precision('high')


# =============================================================================
# 하이퍼파라미터
# =============================================================================
class Config:
    """
    SAC (Discrete) for Atari 하이퍼파라미터.

    D3QN_v3 스타일로 재작성:
      - ALE 환경: ale_py 직접 import (NamespaceNotFound 해결)
      - cleanrl 의존성 제거: 자체 AtariWrapper + ReplayBuffer
      - tyro → argparse
      - 체크포인트/로깅 D3QN_v3 방식
    """

    # Environment
    ENV_NAME   = "ALE/SpaceInvaders-v5"
    FRAME_SKIP = 4      # SAC 원본 논문 기본값
    STACK_SIZE = 4
    IMG_SIZE   = 84
    NO_OP_MAX  = 30

    # Replay Buffer (균등 샘플링)
    REPLAY_CAPACITY = 300_000
    LEARNING_START  = 20_000

    # Training
    BATCH_SIZE      = 64
    DISCOUNT_FACTOR = 0.99
    POLICY_LR       = 3e-4
    Q_LR            = 3e-4
    ADAM_EPS        = 1e-4
    UPDATE_FREQ     = 4          # 4 step마다 1번 업데이트
    TOTAL_FRAMES    = 5_000_000

    # Target Network
    TAU                  = 1.0         # Hard update (tau=1.0)
    TARGET_UPDATE_FREQ   = 8_000       # steps 기준

    # Entropy
    ALPHA               = 0.2
    AUTOTUNE            = True
    TARGET_ENTROPY_SCALE = 0.89       # target_entropy = -scale * log(1/|A|)

    # Reward / Terminal
    REWARD_CLIP           = True
    TERMINAL_ON_LIFE_LOSS = True

    # Evaluation
    EVAL_FREQ     = 250_000          # frames 기준
    EVAL_EPISODES = 5
    EVAL_EPSILON  = 0.001            # 평가 시 그리디에 가깝게

    # Checkpointing & Logging
    CHECKPOINT_FREQ = 250_000        # frames 기준
    CHECKPOINT_DIR  = "./checkpoints_sac"
    LOG_FREQ        = 10             # 에피소드 기준


# =============================================================================
# 1. Atari 환경 래퍼  (D3QN_v3 AtariWrapper 동일)
# =============================================================================
class AtariWrapper(gym.Wrapper):
    """
    Atari 전처리 래퍼.
    - max-pooling on last 2 frames (flickering 제거)
    - random no-op reset
    - grayscale + 84x84 resize
    - frame stacking (4 frames)
    - optional terminal_on_life_loss
    """
    def __init__(self, env, config: Config):
        super().__init__(env)
        self.cfg      = config
        self.frames   = deque(maxlen=config.STACK_SIZE)
        self._obs_buf = np.zeros((2, 210, 160, 3), dtype=np.uint8)
        self.lives    = 0

        low  = np.zeros((config.STACK_SIZE, config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)
        high = np.full_like(low, 255)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.cfg.IMG_SIZE, self.cfg.IMG_SIZE),
                          interpolation=cv2.INTER_AREA)

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        for _ in range(random.randint(1, self.cfg.NO_OP_MAX)):
            frame, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                frame, info = self.env.reset(**kwargs)
        proc = self._preprocess(frame)
        for _ in range(self.cfg.STACK_SIZE):
            self.frames.append(proc)
        self.lives = self.env.unwrapped.ale.lives()
        return self._obs(), info

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False

        for i in range(self.cfg.FRAME_SKIP):
            frame, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if i == self.cfg.FRAME_SKIP - 2:
                self._obs_buf[0] = frame
            if i == self.cfg.FRAME_SKIP - 1:
                self._obs_buf[1] = frame
            if terminated or truncated:
                break

        self.frames.append(self._preprocess(self._obs_buf.max(axis=0)))

        if self.cfg.TERMINAL_ON_LIFE_LOSS:
            cur_lives = self.env.unwrapped.ale.lives()
            if cur_lives < self.lives:
                terminated = True
            self.lives = cur_lives

        clipped_reward = float(np.clip(total_reward, -1.0, 1.0)) if self.cfg.REWARD_CLIP \
                         else total_reward

        return self._obs(), clipped_reward, terminated, truncated, info

    def _obs(self):
        return np.array(self.frames, dtype=np.uint8)


def make_env(config: Config, render=False):
    env = gym.make(
        config.ENV_NAME,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="human" if render else None,
    )
    return AtariWrapper(env, config)


# =============================================================================
# 2. 균등 Replay Buffer  (cleanrl ReplayBuffer 대체)
# =============================================================================
class ReplayBuffer:
    """
    uint8 상태 저장으로 메모리 최적화.
    next_state는 states[(idx+1) % cap] 참조 (D3QN_v3 방식).
    """
    def __init__(self, capacity, state_shape):
        self.capacity    = capacity
        self.state_shape = state_shape

        self.states  = np.zeros((capacity + 1, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.float32)

        self._pos  = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self._pos]     = state
        self.states[self._pos + 1] = next_state
        self.actions[self._pos]    = action
        self.rewards[self._pos]    = reward
        self.dones[self._pos]      = float(done)

        self._pos  = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        idxs      = np.random.randint(0, self._size, size=batch_size)
        next_idxs = idxs + 1
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.states[next_idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self._size


# =============================================================================
# 3. 네트워크 (SAC Discrete: SoftQNetwork + Actor)
# =============================================================================
def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    """
    Soft Q-Network (Critic).
    CNN 인코더는 Actor와 공유하지 않음 (SAC+AE 논문 권장).
    """
    def __init__(self, n_actions, input_shape):
        super().__init__()
        c = input_shape[0]
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(c,  32, kernel_size=8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.inference_mode():
            flat = self.conv(torch.zeros(1, *input_shape)).shape[1]

        self.fc1  = layer_init(nn.Linear(flat, 512))
        self.fc_q = layer_init(nn.Linear(512, n_actions))

    def forward(self, x):
        x = F.relu(self.conv(x.float() / 255.0))
        x = F.relu(self.fc1(x))
        return self.fc_q(x)


class Actor(nn.Module):
    """
    Discrete SAC Actor.
    확률 분포에서 action 샘플링 후 log_prob, action_probs 반환.
    """
    def __init__(self, n_actions, input_shape):
        super().__init__()
        c = input_shape[0]
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(c,  32, kernel_size=8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.inference_mode():
            flat = self.conv(torch.zeros(1, *input_shape)).shape[1]

        self.fc1       = layer_init(nn.Linear(flat, 512))
        self.fc_logits = layer_init(nn.Linear(512, n_actions))

    def forward(self, x):
        x = F.relu(self.conv(x.float() / 255.0))
        x = F.relu(self.fc1(x))
        return self.fc_logits(x)

    def get_action(self, x):
        logits      = self(x)
        dist        = Categorical(logits=logits)
        action      = dist.sample()
        action_probs = dist.probs
        log_prob    = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


# =============================================================================
# 4. SAC 에이전트
# =============================================================================
class SACAgent:
    def __init__(self, env, config: Config, device: torch.device):
        self.cfg          = config
        self.device       = device
        self.n_actions    = env.action_space.n
        self.input_shape  = env.observation_space.shape
        self.total_frames = 0

        # 네트워크 초기화
        self.actor    = Actor(self.n_actions, self.input_shape).to(device)
        self.qf1      = SoftQNetwork(self.n_actions, self.input_shape).to(device)
        self.qf2      = SoftQNetwork(self.n_actions, self.input_shape).to(device)
        self.qf1_tgt  = SoftQNetwork(self.n_actions, self.input_shape).to(device)
        self.qf2_tgt  = SoftQNetwork(self.n_actions, self.input_shape).to(device)
        self.qf1_tgt.load_state_dict(self.qf1.state_dict())
        self.qf2_tgt.load_state_dict(self.qf2.state_dict())

        # Optimizer (eps=1e-4: 수치 안정성)
        self.q_optimizer     = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=config.Q_LR, eps=config.ADAM_EPS,
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.POLICY_LR, eps=config.ADAM_EPS,
        )

        # Automatic entropy tuning
        if config.AUTOTUNE:
            self.target_entropy = (
                -config.TARGET_ENTROPY_SCALE
                * torch.log(1 / torch.tensor(self.n_actions, dtype=torch.float32))
            )
            self.log_alpha  = torch.zeros(1, requires_grad=True, device=device)
            self.alpha      = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=config.Q_LR, eps=config.ADAM_EPS)
        else:
            self.alpha = config.ALPHA

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(config.REPLAY_CAPACITY, self.input_shape)

    # ------------------------------------------------------------------
    def select_action(self, state):
        """학습 중 확률적 행동 선택."""
        s = torch.as_tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.actor.get_action(s)
        return action.item()

    # ------------------------------------------------------------------
    def train_step(self, step):
        """
        SAC 업데이트 한 스텝.
        Returns: (qf_loss, actor_loss, alpha_loss or None)
        """
        if len(self.replay_buffer) < self.cfg.LEARNING_START:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.cfg.BATCH_SIZE)

        s  = torch.as_tensor(states,      dtype=torch.uint8,   device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.uint8,   device=self.device)
        a  = torch.as_tensor(actions,     dtype=torch.long,    device=self.device).unsqueeze(1)
        r  = torch.as_tensor(rewards,     dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(dones,       dtype=torch.float32, device=self.device)

        # ── Critic 업데이트 ──────────────────────────────────────────
        with torch.no_grad():
            _, next_log_pi, next_action_probs = self.actor.get_action(ns)
            qf1_next = self.qf1_tgt(ns)
            qf2_next = self.qf2_tgt(ns)
            min_qf_next = next_action_probs * (
                torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi
            )
            next_q_val = r + (1.0 - d) * self.cfg.DISCOUNT_FACTOR * min_qf_next.sum(dim=1)

        qf1_vals   = self.qf1(s).gather(1, a).view(-1)
        qf2_vals   = self.qf2(s).gather(1, a).view(-1)
        qf1_loss   = F.mse_loss(qf1_vals, next_q_val)
        qf2_loss   = F.mse_loss(qf2_vals, next_q_val)
        qf_loss    = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.q_optimizer.step()

        # ── Actor 업데이트 ───────────────────────────────────────────
        _, log_pi, action_probs = self.actor.get_action(s)
        with torch.no_grad():
            min_qf = torch.min(self.qf1(s), self.qf2(s))
        actor_loss = (action_probs * (self.alpha * log_pi - min_qf)).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Alpha (Entropy) 업데이트 ─────────────────────────────────
        alpha_loss = None
        if self.cfg.AUTOTUNE:
            alpha_loss = (
                action_probs.detach()
                * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())
            ).mean()
            self.a_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ── Target Network Hard Update ────────────────────────────────
        if step % self.cfg.TARGET_UPDATE_FREQ == 0:
            tau = self.cfg.TAU
            for p, tp in zip(self.qf1.parameters(), self.qf1_tgt.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            for p, tp in zip(self.qf2.parameters(), self.qf2_tgt.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return qf_loss.item(), actor_loss.item(), alpha_loss.item() if alpha_loss else None

    # ------------------------------------------------------------------
    def save(self, path, extra=None):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        ckpt = {
            'frame':        self.total_frames,
            'actor':        self.actor.state_dict(),
            'qf1':          self.qf1.state_dict(),
            'qf2':          self.qf2.state_dict(),
            'qf1_tgt':      self.qf1_tgt.state_dict(),
            'qf2_tgt':      self.qf2_tgt.state_dict(),
            'q_optimizer':  self.q_optimizer.state_dict(),
            'act_optimizer':self.actor_optimizer.state_dict(),
            'alpha':        self.alpha,
        }
        if self.cfg.AUTOTUNE:
            ckpt['log_alpha']   = self.log_alpha.data
            ckpt['a_optimizer'] = self.a_optimizer.state_dict()
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f'  [Saved] {path}')

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.qf1.load_state_dict(ckpt['qf1'])
        self.qf2.load_state_dict(ckpt['qf2'])
        self.qf1_tgt.load_state_dict(ckpt['qf1_tgt'])
        self.qf2_tgt.load_state_dict(ckpt['qf2_tgt'])
        self.q_optimizer.load_state_dict(ckpt['q_optimizer'])
        self.actor_optimizer.load_state_dict(ckpt['act_optimizer'])
        self.alpha       = ckpt.get('alpha', self.cfg.ALPHA)
        self.total_frames = ckpt.get('frame', 0)
        if self.cfg.AUTOTUNE and 'log_alpha' in ckpt:
            self.log_alpha.data = ckpt['log_alpha']
            self.a_optimizer.load_state_dict(ckpt['a_optimizer'])
        print(f'  [Loaded] {path}  (frame={self.total_frames:,})')


# =============================================================================
# 5. 평가 루프
# =============================================================================
def evaluate(agent: SACAgent, config: Config, n_episodes=None):
    """Reward clipping 없는 별도 환경에서 실제 게임 점수 수집."""
    n_episodes = n_episodes or config.EVAL_EPISODES

    eval_cfg = Config()
    eval_cfg.REWARD_CLIP           = False
    eval_cfg.TERMINAL_ON_LIFE_LOSS = False
    env = make_env(eval_cfg)

    agent.actor.eval()
    scores = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done  = False
        score = 0.0
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            score += reward
        scores.append(score)

    env.close()
    agent.actor.train()
    return float(np.mean(scores)), float(np.std(scores))


# =============================================================================
# 6. 학습 루프
# =============================================================================
def train(config: Config, resume_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = f"{config.ENV_NAME.replace('/', '_')}__sac__{int(time.time())}"

    print(f'Device: {device}')
    print(f'Training SAC (Discrete) on {config.ENV_NAME}')
    print(f'  Policy LR          : {config.POLICY_LR}')
    print(f'  Q LR               : {config.Q_LR}')
    print(f'  Batch Size         : {config.BATCH_SIZE}')
    print(f'  Replay Capacity    : {config.REPLAY_CAPACITY:,}')
    print(f'  Learning Start     : {config.LEARNING_START:,}')
    print(f'  Update Freq        : every {config.UPDATE_FREQ} steps')
    print(f'  Target Update      : every {config.TARGET_UPDATE_FREQ} steps (tau={config.TAU})')
    print(f'  Autotune Entropy   : {config.AUTOTUNE}  (scale={config.TARGET_ENTROPY_SCALE})')
    print(f'  Frame Skip         : {config.FRAME_SKIP}')
    print(f'  Reward Clip        : {config.REWARD_CLIP}')
    print(f'  TerminalOnLifeLoss : {config.TERMINAL_ON_LIFE_LOSS}')
    print(f'  Total Frames       : {config.TOTAL_FRAMES:,}')
    print()

    writer = SummaryWriter(f"runs/{run_name}")
    env    = make_env(config)
    agent  = SACAgent(env, config, device)

    if resume_path:
        agent.load(resume_path)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    episode_rewards = deque(maxlen=100)
    qf_losses       = deque(maxlen=500)
    actor_losses    = deque(maxlen=500)
    best_eval_score = -float('inf')
    step = episode  = 0
    start_time      = time.time()

    state, _ = env.reset()
    ep_reward = 0.0

    print('Starting training...')

    while agent.total_frames < config.TOTAL_FRAMES:

        # 행동 선택: 학습 시작 전엔 랜덤
        if agent.total_frames < config.LEARNING_START:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        agent.total_frames += config.FRAME_SKIP
        step      += 1
        ep_reward += reward

        # UPDATE_FREQ마다 학습
        if step % config.UPDATE_FREQ == 0:
            result = agent.train_step(step)
            if result is not None:
                qf_loss, actor_loss, alpha_loss = result
                qf_losses.append(qf_loss)
                actor_losses.append(actor_loss)
                # TensorBoard 로깅
                if step % 100 == 0:
                    writer.add_scalar("losses/qf_loss",    qf_loss,    agent.total_frames)
                    writer.add_scalar("losses/actor_loss", actor_loss, agent.total_frames)
                    writer.add_scalar("losses/alpha",      agent.alpha, agent.total_frames)
                    if alpha_loss is not None:
                        writer.add_scalar("losses/alpha_loss", alpha_loss, agent.total_frames)
                    fps = agent.total_frames / (time.time() - start_time + 1e-8)
                    writer.add_scalar("charts/SPS", fps, agent.total_frames)

        # Checkpoint
        if agent.total_frames % config.CHECKPOINT_FREQ == 0:
            path = os.path.join(config.CHECKPOINT_DIR, f'sac_frame_{agent.total_frames}.pth')
            agent.save(path)

        # Evaluation
        if agent.total_frames % config.EVAL_FREQ == 0 \
                and len(agent.replay_buffer) >= config.LEARNING_START:
            eval_mean, eval_std = evaluate(agent, config)
            print(f'  >> Eval @ frame {agent.total_frames:,}: '
                  f'score = {eval_mean:.1f} +/- {eval_std:.1f}')
            writer.add_scalar("charts/eval_score", eval_mean, agent.total_frames)
            if eval_mean > best_eval_score:
                best_eval_score = eval_mean
                best_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                agent.save(best_path, extra={'eval_score': eval_mean})

        # Episode end
        if done:
            episode += 1
            episode_rewards.append(ep_reward)
            writer.add_scalar("charts/episodic_return", ep_reward, agent.total_frames)

            if episode % config.LOG_FREQ == 0:
                elapsed = time.time() - start_time
                fps     = agent.total_frames / (elapsed + 1e-8)
                avg_r   = np.mean(episode_rewards)
                avg_qf  = np.mean(qf_losses)  if qf_losses  else 0.0
                avg_act = np.mean(actor_losses) if actor_losses else 0.0
                status  = 'filling' if len(agent.replay_buffer) < config.LEARNING_START \
                          else 'training'
                print(
                    f'Frame: {agent.total_frames:>10,}/{config.TOTAL_FRAMES:,} | '
                    f'Ep: {episode:>5} | '
                    f'Reward: {ep_reward:>7.1f} | '
                    f'Avg(100): {avg_r:>7.1f} | '
                    f'QF_Loss: {avg_qf:.4f} | '
                    f'Act_Loss: {avg_act:.4f} | '
                    f'Alpha: {agent.alpha:.4f} | '
                    f'Buffer: {len(agent.replay_buffer):>7,} | '
                    f'FPS: {fps:>6.1f} | '
                    f'{status}'
                )

            state, _ = env.reset()
            ep_reward = 0.0

    agent.save(os.path.join(config.CHECKPOINT_DIR, 'final_model.pth'))
    env.close()
    writer.close()
    print('Training complete.')


# =============================================================================
# 7. 테스트
# =============================================================================
def test(model_path, config: Config, n_episodes=10, render=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_cfg = Config()
    test_cfg.REWARD_CLIP           = False
    test_cfg.TERMINAL_ON_LIFE_LOSS = False
    env = make_env(test_cfg, render=render)

    agent = SACAgent(env, config, device)
    agent.load(model_path)
    agent.actor.eval()

    scores = []
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        done  = False
        score = 0.0
        while not done:
            action = agent.select_action(state)
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

    print(f'\nBenchmark (SpaceInvaders):')
    print(f'  Random : ~148')
    print(f'  DQN    : ~1,976')
    print(f'  SAC    : ~1,151 (CleanRL 기준)')
    print(f'  Ours   : {np.mean(scores):.0f}')


# =============================================================================
# 8. 진입점
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SAC (Discrete) for Atari — D3QN_v3 스타일',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sac_atari_v2.py --train
  python sac_atari_v2.py --train --resume checkpoints_sac/sac_frame_1000000.pth
  python sac_atari_v2.py --test  --model  checkpoints_sac/best_model.pth
  python sac_atari_v2.py --test  --model  checkpoints_sac/best_model.pth --render --episodes 20
  python sac_atari_v2.py --train --env ALE/Pong-v5
        """,
    )
    parser.add_argument('--train',        action='store_true')
    parser.add_argument('--test',         action='store_true')
    parser.add_argument('--model',        type=str,   default=None)
    parser.add_argument('--resume',       type=str,   default=None)
    parser.add_argument('--render',       action='store_true')
    parser.add_argument('--episodes',     type=int,   default=10)
    parser.add_argument('--env',          type=str,   default=None,  help='환경 ID (기본: ALE/SpaceInvaders-v5)')
    parser.add_argument('--lr',           type=float, default=None,  help='Q/Policy 공통 LR')
    parser.add_argument('--total-frames', type=int,   default=None)
    parser.add_argument('--no-autotune', action='store_true',        help='엔트로피 자동 튜닝 비활성화')
    parser.add_argument('--alpha',        type=float, default=None,  help='고정 alpha (--no-autotune 시 사용)')

    args   = parser.parse_args()
    config = Config()

    if args.env:          config.ENV_NAME   = args.env
    if args.lr:           config.POLICY_LR  = config.Q_LR = args.lr
    if args.total_frames: config.TOTAL_FRAMES = args.total_frames
    if args.no_autotune:  config.AUTOTUNE   = False
    if args.alpha:        config.ALPHA       = args.alpha

    if args.train:
        train(config, resume_path=args.resume)
    elif args.test:
        if not args.model:
            parser.error('--test requires --model MODEL_PATH')
        test(args.model, config, n_episodes=args.episodes, render=args.render)
    else:
        parser.print_help()
