"""
D3QN_v3 (Dueling Double DQN + Prioritized Experience Replay)
for ALE/SpaceInvaders-v5

Base papers:
  [1] Mnih et al. (2013)        - DQN
  [2] van Hasselt et al. (2015) - Double DQN
  [3] Wang et al. (2016)        - Dueling Network
  [4] Schaul et al. (2015)      - Prioritized Experience Replay (PER)

v3 vs v2 버그 수정 / 개선 요약:
  [BUG FIX 1] TRAIN_FREQ: 1 → 4
              Mnih 2013 원본 및 Wang 2016 설정. v2는 매 스텝 학습으로
              gradient 업데이트가 16배 과도해 Loss 폭발의 직접 원인.

  [BUG FIX 2] LEARNING_RATE: 2.5e-4 → 6.25e-5
              Wang et al. 2016 Dueling 논문 권장값.
              PER은 중요 샘플을 집중 샘플링 → LR이 높으면 발산.

  [BUG FIX 3] next_state 저장 버그 수정
              v2: states[(pos+1)%cap]에 미리 쓰기 → wrap-around 시 데이터 오염.
              v3: next_states 전용 배열 사용 (메모리 ~7GB, 정확성 우선).

  [BUG FIX 4] gradient rescaling 위치 수정
              v2: features[-2] (3번째 Conv)에 훅 → 잘못된 위치.
              v3: flatten 직후 별도 모듈로 정확히 적용 [3].

  [BUG FIX 5] Optimizer: RMSprop → Adam
              Wang et al. 2016 Dueling 논문은 Adam 사용.
              Adam이 PER와 더 안정적으로 조합됨.

  [BUG FIX 6] EPSILON_END: 0.1 → 0.01
              탐색 종료 후 10% 랜덤은 후반 최적 학습 방해.
              논문 표준값 0.01 적용.

  [BUG FIX 7] SOFT_UPDATE: 기본값 False 명시 (PER와 조합 시 Loss 폭발)

Usage:
    python D3QN_v3.py --train
    python D3QN_v3.py --train --resume checkpoints_v3/d3qn_frame_1000000.pth
    python D3QN_v3.py --test  --model  checkpoints_v3/best_model.pth
    python D3QN_v3.py --test  --model  checkpoints_v3/best_model.pth --render --episodes 20
    python D3QN_v3.py --train --life-loss
"""

import os
import sys
import random
import argparse
import time
import csv
from collections import deque

import numpy as np
import cv2
from PIL import Image

import gymnasium as gym
import ale_py  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.stdout.reconfigure(encoding='utf-8')

# TF32 가속 (Ampere+ GPU: RTX 30xx/40xx)
torch.set_float32_matmul_precision('high')


# =============================================================================
# 하이퍼파라미터
# =============================================================================
class Config:
    """
    D3QN + PER 하이퍼파라미터.

    v3 핵심 변경:
      TRAIN_FREQ    : 1 → 4       (Mnih 2013 원본, Loss 폭발 방지)
      LEARNING_RATE : 2.5e-4 → 6.25e-5  (Wang 2016 Dueling 논문 권장값)
      EPSILON_END   : 0.1 → 0.01  (표준값)
      SOFT_UPDATE   : False        (PER와 조합 시 발산 방지)
      Optimizer     : Adam         (Wang 2016 논문 사용 optimizer)
    """

    # Environment
    ENV_NAME   = "ALE/SpaceInvaders-v5"
    FRAME_SKIP = 3      # k=3 for SpaceInvaders (k=4 makes lasers invisible) [1]
    STACK_SIZE = 4
    IMG_SIZE   = 84
    NO_OP_MAX  = 30

    # Replay Buffer
    # [MEM FIX] 1M → 500K: 메모리 절감 (~14 GB, 기존 next_states 제거로 추가 절감)
    # 500K도 논문 수준 성능 달성에 충분 (레퍼런스 ver3: 100K로도 학습됨)
    # 메모리 계산 (next_states 배열 없이 states만):
    #   200K:   5.3 GB
    #   500K:  13.1 GB  ← PER 논문 (Schaul et al. 2015) 권장값
    #   1000K: 26.3 GB  ← Mnih 2013 원본값, 64GB RAM 환경
    REPLAY_CAPACITY = 1_000_000
    LEARNING_START  = 50_000

    # PER 파라미터 [4]
    PER_ALPHA       = 0.6          # 우선순위 지수 (0=균등, 1=완전 우선순위)
    PER_BETA_START  = 0.4          # IS weight 보정 초기값
    PER_BETA_FRAMES = 50_000_000   # beta가 1.0에 도달하는 frame 수 (=TOTAL_FRAMES)
    PER_EPS         = 1e-6         # priority 최솟값 (0 방지)

    # Training
    BATCH_SIZE      = 32
    DISCOUNT_FACTOR = 0.99
    LEARNING_RATE   = 6.25e-5     # [FIX 2] Wang et al. 2016 권장값 (v2: 2.5e-4)
    ADAM_EPS        = 1.5e-4      # Adam epsilon (Wang et al. 2016)
    TRAIN_FREQ      = 4           # [FIX 1] 4 step마다 1번 학습 (v2: 1, Mnih 2013 원본)
    TOTAL_FRAMES    = 70_000_000

    # Target Network
    # [FIX 7] SOFT_UPDATE=False: PER와 soft update 조합 시 Loss 폭발 확인
    # 원인: PER가 TD error 큰 것 집중 샘플링 + soft update로 target 계속 변함
    #       → 양성 피드백 → Loss 발산
    SOFT_UPDATE        = False     # PER와 함께 쓸 때 반드시 False
    TAU                = 0.005     # SOFT_UPDATE=True 일 때만 사용
    TARGET_UPDATE_FREQ = 10_000   # Hard update 주기 (steps)

    # Exploration
    EPSILON_START = 1.0
    EPSILON_END   = 0.01          # [FIX 6] 0.1 → 0.01 (표준값, 후반 탐색 억제)
    EPSILON_DECAY = 1_000_000

    # Gradient & Reward
    GRAD_CLIP_NORM = 10.0
    REWARD_CLIP    = True

    TERMINAL_ON_LIFE_LOSS = False

    # Evaluation
    EVAL_FREQ     = 250_000
    EVAL_EPISODES = 5
    EVAL_EPSILON  = 0.001

    # Checkpointing & Logging
    CHECKPOINT_FREQ = 250_000
    CHECKPOINT_DIR  = "./checkpoints_v3_logs"
    LOG_FREQ        = 10

    # Action Logging (매 스텝 이미지 + CSV 저장)
    LOG_ACTIONS     = False           # --log-actions 플래그로 활성화
    LOG_ACTIONS_DIR = "./action_logs" # --log-dir 로 변경 가능


# =============================================================================
# 1. Atari 환경 래퍼
# =============================================================================
class AtariWrapper(gym.Wrapper):
    """
    Atari 전처리 래퍼.
    - max-pooling on last 2 frames (flickering 제거) [1]
    - random no-op reset [1]
    - grayscale + 84x84 resize [1]
    - frame stacking (4 frames) [1]
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
# 1-b. Action Logger (매 스텝 이미지 + 메타데이터 CSV 저장)
# =============================================================================
class ActionLogger:
    """
    actions_to_csv.py 스타일로 에이전트의 매 action 선택 시 정보를 저장.

    저장 항목:
      - 전처리 전 원본 RGB 프레임 (PNG 이미지)
      - episode, step_in_episode, global_step (= total_frames)
      - action, reward, terminated, truncated
      - epsilon (현재 탐색률)
      - q_values (각 action의 Q값, 콤마 구분)
      - selected_q (선택된 action의 Q값)
      - loss (직전 학습 loss, 없으면 빈칸)
      - image_path (저장된 이미지 경로)

    Usage:
        logger = ActionLogger(base_dir="./action_logs", mode="train")
        logger.log_step(frame, episode, step_in_ep, global_step,
                        action, reward, terminated, truncated,
                        epsilon=0.1, q_values=[1.2, 0.3, ...], loss=0.05)
        logger.close()
    """

    CSV_HEADER = [
        "episode",
        "step_in_episode",
        "global_step",
        "action",
        "reward",
        "terminated",
        "truncated",
        "epsilon",
        "q_values",
        "selected_q",
        "loss",
        "image_path",
    ]

    def __init__(self, base_dir: str, mode: str = "train"):
        """
        Args:
            base_dir: 로그 저장 최상위 디렉토리
            mode: "train" 또는 "test" (하위 폴더 구분)
        """
        self.base_dir  = os.path.join(base_dir, mode)
        self.image_dir = os.path.join(self.base_dir, "images")
        self.csv_path  = os.path.join(self.base_dir, "actions.csv")

        os.makedirs(self.image_dir, exist_ok=True)

        # CSV 파일이 없으면 헤더 작성
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.CSV_HEADER)

        # 성능을 위해 CSV 파일을 열어놓고 사용
        self._csv_file   = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._flush_counter = 0

        print(f"  [ActionLogger] Saving to: {self.base_dir}")
        print(f"  [ActionLogger] CSV: {self.csv_path}")
        print(f"  [ActionLogger] Images: {self.image_dir}")

    def log_step(
        self,
        frame: np.ndarray,
        episode: int,
        step_in_episode: int,
        global_step: int,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        epsilon: float = 0.0,
        q_values: np.ndarray = None,
        loss: float = None,
    ):
        """
        단일 스텝 정보를 이미지 + CSV로 저장.

        Args:
            frame: 전처리 전 상태 (stacked uint8 ndarray, shape=(4,84,84))
                   또는 원본 RGB 프레임.
                   stacked frame인 경우 마지막 프레임만 저장.
            episode: 현재 에피소드 번호
            step_in_episode: 에피소드 내 스텝 번호
            global_step: 전체 프레임 수 (total_frames)
            action: 선택된 action 인덱스
            reward: 받은 reward
            terminated: 환경 terminated 플래그
            truncated: 환경 truncated 플래그
            epsilon: 현재 epsilon 값
            q_values: 전체 action의 Q값 배열 (optional)
            loss: 직전 학습 loss (optional)
        """
        # 이미지 저장 (stacked frame이면 마지막 프레임만)
        image_filename = f"ep{episode:04d}_step{step_in_episode:05d}_g{global_step:08d}.png"
        image_path = os.path.join(self.image_dir, image_filename)

        if frame.ndim == 3 and frame.shape[0] == 4:
            # stacked grayscale (4, 84, 84) → 마지막 프레임
            save_frame = frame[-1]  # (84, 84)
        elif frame.ndim == 2:
            # 단일 grayscale (84, 84)
            save_frame = frame
        else:
            # RGB 등 그대로 저장
            save_frame = frame

        Image.fromarray(save_frame).save(image_path)

        # Q-values 문자열 변환
        q_str = ""
        selected_q = ""
        if q_values is not None:
            q_str = ";".join(f"{v:.4f}" for v in q_values)
            if 0 <= action < len(q_values):
                selected_q = f"{q_values[action]:.4f}"

        loss_str = f"{loss:.6f}" if loss is not None else ""

        # CSV 기록
        self._csv_writer.writerow([
            episode,
            step_in_episode,
            global_step,
            action,
            reward,
            int(terminated),
            int(truncated),
            f"{epsilon:.6f}",
            q_str,
            selected_q,
            loss_str,
            image_path,
        ])

        # 100 스텝마다 flush (디스크 쓰기 최적화)
        self._flush_counter += 1
        if self._flush_counter % 100 == 0:
            self._csv_file.flush()

    def close(self):
        """CSV 파일 닫기."""
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
            print(f"  [ActionLogger] Closed. Total logged: {self._flush_counter} steps")


# =============================================================================
# 2. Dueling DQN 네트워크
# =============================================================================
class GradRescale(nn.Module):
    """
    [FIX 4] gradient rescaling 1/sqrt(2) 전용 모듈.

    Wang et al. 2016 §4: "we rescale the combined gradient entering
    the last convolutional layer by 1/sqrt(2)"
    → flatten 직후, 두 stream에 입력되기 직전에 적용해야 함.

    v2 버그: self.features[-2].register_full_backward_hook()으로
             3번째 Conv에 걸었으나 잘못된 위치.
    v3 수정: 별도 nn.Module로 분리하여 forward/backward 모두 정확히 처리.
    """
    SCALE = 1.0 / (2.0 ** 0.5)

    def forward(self, x):
        return x

    def backward_hook(self, module, grad_input, grad_output):
        return tuple(
            g * self.SCALE if g is not None else None
            for g in grad_input
        )


class DuelingDQN(nn.Module):
    """
    Dueling Network Architecture [3].
    CNN [1] + Value/Advantage streams + gradient rescaling 1/sqrt(2) [3]

    [FIX 4] gradient rescaling을 flatten 직후 GradRescale 모듈로 정확히 적용.
    [FIX 5] Optimizer는 외부에서 Adam으로 생성 (Wang et al. 2016).
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c,  32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        with torch.no_grad():
            flat = self.features(torch.zeros(1, *input_shape)).flatten().shape[0]

        # [FIX 4] flatten 직후 grad rescaling 모듈 삽입
        self.grad_rescale = GradRescale()
        self.grad_rescale.register_full_backward_hook(self.grad_rescale.backward_hook)

        self.value_stream = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

        # Kaiming 초기화
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x         = x.contiguous().float().mul_(1.0 / 255.0)
        features  = self.features(x).flatten(1)
        features  = self.grad_rescale(features)   # [FIX 4] 올바른 위치
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# =============================================================================
# 3. SumTree  [4]
# =============================================================================
class SumTree:
    """
    PER의 핵심 자료구조. O(log n) 샘플링 / 업데이트.

    구조:
        - 내부 이진 트리: 리프 노드가 각 transition의 priority
        - 부모 노드는 자식들의 priority 합
        - 루트 노드 = 전체 priority 합 → 균등 구간으로 나눠 O(log n) 샘플링
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.pos      = 0
        self.size     = 0

    def _propagate(self, idx, delta):
        """리프에서 루트까지 변화량 전파. 반복문으로 스택 오버플로 방지 및 속도 개선."""
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def update(self, idx, priority):
        """인덱스 idx의 priority를 업데이트."""
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority):
        """새 priority 추가. 리프 인덱스 반환."""
        leaf_idx = self.pos + self.capacity - 1
        self.update(leaf_idx, priority)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return leaf_idx

    def get(self, value):
        """
        value에 해당하는 리프를 O(log n)으로 탐색.
        value: [0, total_priority) 범위의 실수
        반환: (leaf_idx, priority, data_idx)
        """
        idx = 0
        while True:
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], data_idx

    @property
    def total(self):
        return self.tree[0]

    @property
    def max_priority(self):
        """현재 저장된 리프 중 최대 priority."""
        return self.tree[self.capacity - 1:self.capacity - 1 + self.size].max()


# =============================================================================
# 4. PrioritizedReplayBuffer  [4]
# =============================================================================
class PrioritizedReplayBuffer:
    """
    SumTree 기반 Prioritized Experience Replay 버퍼.

    IS weight 공식 [4] Eq.1:
        w_i = (N * P(i))^(-beta)  → 최대값으로 정규화

    메모리 최적화 (v4):
        - states 배열만 사용: capacity × 4 × 84 × 84 × uint8
        - next_state는 states[(idx+1)%cap]으로 참조 (v2 방식)
        - 단, done=True인 transition은 episode 경계이므로
          next_state가 의미 없음 (done 마스킹으로 TD target에서 제거됨)
        - 따라서 wrap-around 오염은 학습에 영향 없음:
            done=True  → target_q = r + 0 (next_q 무시)
            done=False → states[(idx+1)%cap] 참조 (정상 케이스)
        - 200K 기준: 5.3 GB (next_states 배열 없이 절반)
    """
    def __init__(self, capacity, state_shape, alpha):
        self.capacity    = capacity
        self.alpha       = alpha
        self.state_shape = state_shape

        self.tree = SumTree(capacity)

        # states만 사용, next_states 배열 없음 (메모리 절반)
        self.states  = np.zeros((capacity + 1, *state_shape), dtype=np.uint8)
        # +1: 마지막 슬롯이 wrap-around 없이 next_state를 참조할 수 있도록
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.float32)

        self._pos = 0

    def push(self, state, action, reward, next_state, done):
        """새 transition 저장. 초기 priority = 현재 max (첫 번째는 1.0)."""
        max_p    = self.tree.max_priority if self.tree.size > 0 else 1.0
        priority = max_p ** self.alpha

        self.states[self._pos]             = state
        self.states[self._pos + 1]         = next_state  # +1 슬롯에 미리 저장
        self.actions[self._pos]            = action
        self.rewards[self._pos]            = reward
        self.dones[self._pos]              = float(done)

        self.tree.add(priority)
        # wrap-around: capacity에 도달하면 0으로 돌아가되
        # states는 capacity+1 크기라 +1 슬롯 참조가 항상 안전
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        """priority에 비례하여 batch_size개 샘플링."""
        total    = self.tree.total
        segment  = total / batch_size

        leaf_idxs  = []
        data_idxs  = []
        priorities = []

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            v  = random.uniform(lo, hi)
            leaf_idx, p, data_idx = self.tree.get(v)
            leaf_idxs.append(leaf_idx)
            data_idxs.append(data_idx)
            priorities.append(p)

        data_idxs  = np.array(data_idxs)
        # next_state: 같은 인덱스의 +1 슬롯 참조 (states는 capacity+1 크기)
        next_idxs  = data_idxs + 1
        priorities = np.array(priorities, dtype=np.float64)

        # IS weight: w_i = (N * P(i))^(-beta), 최대값 정규화
        N          = self.tree.size
        probs      = priorities / (total + 1e-8)
        is_weights = (N * probs) ** (-beta)
        is_weights = (is_weights / is_weights.max()).astype(np.float32)

        return (
            self.states[data_idxs],
            self.actions[data_idxs],
            self.rewards[data_idxs],
            self.states[next_idxs],   # +1 슬롯 참조 (wrap-around 없음)
            self.dones[data_idxs],
            leaf_idxs,
            is_weights,
        )

    def update_priorities(self, leaf_idxs, td_errors, eps):
        """TD error 기반으로 priority 갱신."""
        for idx, err in zip(leaf_idxs, td_errors):
            p = (abs(err) + eps) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.size


# =============================================================================
# 5. D3QN + PER 에이전트
# =============================================================================
class D3QNAgent:
    def __init__(self, env, config: Config, device: torch.device):
        self.cfg          = config
        self.device       = device
        self.n_actions    = env.action_space.n
        self.input_shape  = env.observation_space.shape
        self.total_frames = 0

        self.q_network      = DuelingDQN(self.input_shape, self.n_actions).to(device)
        self.target_network = DuelingDQN(self.input_shape, self.n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # torch.compile (backend=eager: Windows Triton 미지원 대응)
        if hasattr(torch, 'compile'):
            try:
                self.q_network      = torch.compile(self.q_network,      backend='eager')
                self.target_network = torch.compile(self.target_network, backend='eager')
                print('  [torch.compile] applied (backend=eager)')
            except Exception as e:
                print(f'  [torch.compile] skipped: {e}')

        # [FIX 5] Adam optimizer (Wang et al. 2016 Dueling 논문 사용 optimizer)
        # v2에서 사용한 RMSprop은 Mnih 2013 원본용; Dueling/PER 조합엔 Adam이 더 안정적
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.LEARNING_RATE,
            eps=config.ADAM_EPS,
        )

        self.replay_buffer = PrioritizedReplayBuffer(
            config.REPLAY_CAPACITY,
            self.input_shape,
            config.PER_ALPHA,
        )

    def get_epsilon(self):
        frac = min(1.0, self.total_frames / self.cfg.EPSILON_DECAY)
        return self.cfg.EPSILON_START - frac * (self.cfg.EPSILON_START - self.cfg.EPSILON_END)

    def get_beta(self):
        """beta를 PER_BETA_START에서 1.0까지 선형 annealing."""
        frac = min(1.0, self.total_frames / self.cfg.PER_BETA_FRAMES)
        return self.cfg.PER_BETA_START + frac * (1.0 - self.cfg.PER_BETA_START)

    def select_action(self, state, epsilon=None, return_q=False):
        eps = epsilon if epsilon is not None else self.get_epsilon()
        if random.random() < eps:
            action = random.randrange(self.n_actions)
            if return_q:
                # 랜덤 action이지만 Q값은 참고용으로 계산
                s = torch.as_tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = self.q_network(s).squeeze(0).cpu().numpy()
                return action, q_vals
            return action
        s = torch.as_tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_out = self.q_network(s)
            action = q_out.argmax().item()
            if return_q:
                return action, q_out.squeeze(0).cpu().numpy()
            return action

    def train_step(self):
        """
        Double DQN update [2] + PER IS weight 보정 [4].

        Double DQN target:
            a* = argmax_a Q(s', a; theta)         ← online 네트워크가 action 선택
            y  = r + gamma * Q(s', a*; theta-)    ← target 네트워크가 Q값 평가

        PER 보정:
            L = mean(w_i * huber(Q(s,a) - y_i))   ← IS weight로 편향 보정
        """
        if len(self.replay_buffer) < self.cfg.LEARNING_START:
            return None

        beta = self.get_beta()
        states, actions, rewards, next_states, dones, leaf_idxs, is_weights = \
            self.replay_buffer.sample(self.cfg.BATCH_SIZE, beta)

        s  = torch.as_tensor(states,      dtype=torch.uint8,   device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.uint8,   device=self.device)
        a  = torch.as_tensor(actions,     dtype=torch.long,    device=self.device).unsqueeze(1)
        r  = torch.as_tensor(rewards,     dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(dones,       dtype=torch.float32, device=self.device)
        w  = torch.as_tensor(is_weights,  dtype=torch.float32, device=self.device)

        current_q = self.q_network(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            # Double DQN: online으로 action 선택, target으로 Q값 평가
            next_a   = self.q_network(ns).argmax(1, keepdim=True)
            next_q   = self.target_network(ns).gather(1, next_a).squeeze(1)
            target_q = r + self.cfg.DISCOUNT_FACTOR * next_q * (1.0 - d)

        # TD error (priority 업데이트용)
        td_errors = (target_q - current_q).detach().cpu().numpy()

        # IS weight 적용 weighted Huber loss
        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (w * elementwise_loss).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.cfg.GRAD_CLIP_NORM)
        self.optimizer.step()

        # priority 갱신
        self.replay_buffer.update_priorities(leaf_idxs, td_errors, self.cfg.PER_EPS)

        return loss.item()

    def update_target(self, soft=False, tau=0.005):
        """
        Hard update (soft=False): target 가중치를 online으로 통째로 복사.
        Soft update (soft=True) : target = tau * online + (1-tau) * target
                                  (SOFT_UPDATE=False 권장 - PER와 조합 시 발산)
        """
        if soft:
            for t_param, q_param in zip(
                    self.target_network.parameters(),
                    self.q_network.parameters()):
                t_param.data.copy_(tau * q_param.data + (1.0 - tau) * t_param.data)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path, extra=None):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        ckpt = {
            'frame':     self.total_frames,
            'q_network': self.q_network.state_dict(),
            'target':    self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon':   self.get_epsilon(),
            'beta':      self.get_beta(),
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f'  [Saved] {path}')

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(ckpt['q_network'])
        self.target_network.load_state_dict(ckpt.get('target', ckpt['q_network']))
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_frames = ckpt.get('frame', 0)
        print(f'  [Loaded] {path}  (frame={self.total_frames:,})')


# =============================================================================
# 6. 평가 루프
# =============================================================================
def evaluate(agent: D3QNAgent, config: Config, n_episodes=None):
    """Reward clipping 없는 별도 환경에서 실제 게임 점수 수집."""
    n_episodes = n_episodes or config.EVAL_EPISODES

    eval_cfg = Config()
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


# =============================================================================
# 7. 학습 루프
# =============================================================================
def train(config: Config, resume_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Training D3QN+PER (v3) on {config.ENV_NAME}')
    print(f'  Learning Rate      : {config.LEARNING_RATE}  (v2: 2.5e-4 → v3: 6.25e-5)')
    print(f'  Optimizer          : Adam (eps={config.ADAM_EPS})  (v2: RMSprop)')
    print(f'  Train Freq         : every {config.TRAIN_FREQ} steps  (v2: every 1 step)')
    print(f'  Replay Buffer      : {config.REPLAY_CAPACITY:,}  [PER, SumTree, uint8, ~{config.REPLAY_CAPACITY*4*84*84/1024**3:.1f}GB]')
    print(f'  PER alpha          : {config.PER_ALPHA}')
    print(f'  PER beta           : {config.PER_BETA_START} -> 1.0 over {config.PER_BETA_FRAMES:,} frames')
    print(f'  PER eps            : {config.PER_EPS}')
    print(f'  Learning Start     : {config.LEARNING_START:,} transitions')
    print(f'  Target Update      : {"Soft (tau=" + str(config.TAU) + ", every step)" if config.SOFT_UPDATE else "Hard (every " + str(config.TARGET_UPDATE_FREQ) + " steps)"}')
    print(f'  Epsilon            : {config.EPSILON_START} -> {config.EPSILON_END} over {config.EPSILON_DECAY:,} frames  (v2 end: 0.1 → v3: 0.01)')
    print(f'  Frame Skip         : {config.FRAME_SKIP}  (SpaceInvaders k=3)')
    print(f'  Reward Clip        : {config.REWARD_CLIP}')
    print(f'  TerminalOnLifeLoss : {config.TERMINAL_ON_LIFE_LOSS}')
    print(f'  Total Frames       : {config.TOTAL_FRAMES:,}')
    print(f'  Log Actions        : {config.LOG_ACTIONS}')
    print()

    env   = make_env(config)
    agent = D3QNAgent(env, config, device)

    if resume_path:
        agent.load(resume_path)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Action Logger 초기화
    action_logger = None
    if config.LOG_ACTIONS:
        action_logger = ActionLogger(base_dir=config.LOG_ACTIONS_DIR, mode="train")

    episode_rewards = deque(maxlen=100)
    losses          = deque(maxlen=500)
    best_eval_score = -float('inf')
    step = episode  = 0
    start_time      = time.time()
    last_loss       = None  # 직전 학습 loss (로깅용)

    state, _  = env.reset()
    ep_reward = 0.0
    step_in_episode = 0

    print('Starting training...')

    while agent.total_frames < config.TOTAL_FRAMES:

        # action 선택 (LOG_ACTIONS면 Q-values도 함께 반환)
        if action_logger is not None:
            action, q_values = agent.select_action(state, return_q=True)
        else:
            action = agent.select_action(state)
            q_values = None

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)

        agent.total_frames += config.FRAME_SKIP
        step      += 1
        step_in_episode += 1
        ep_reward += reward

        # [FIX 1] TRAIN_FREQ=4: 4 step마다 1번 학습 (v2: 매 step)
        if step % config.TRAIN_FREQ == 0:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
                last_loss = loss

        # Action 로깅 (매 스텝)
        if action_logger is not None:
            action_logger.log_step(
                frame=state,
                episode=episode + 1,
                step_in_episode=step_in_episode,
                global_step=agent.total_frames,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                epsilon=agent.get_epsilon(),
                q_values=q_values,
                loss=last_loss,
            )

        state = next_state

        # Target update
        if config.SOFT_UPDATE:
            agent.update_target(soft=True, tau=config.TAU)
        elif step % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target(soft=False)
            print(f'  [Target Hard Updated] frame={agent.total_frames:,}')

        # Checkpoint
        if agent.total_frames % config.CHECKPOINT_FREQ == 0:
            path = os.path.join(config.CHECKPOINT_DIR, f'd3qn_frame_{agent.total_frames}.pth')
            agent.save(path)

        # Evaluation
        if agent.total_frames % config.EVAL_FREQ == 0 \
                and len(agent.replay_buffer) >= config.LEARNING_START:
            eval_mean, eval_std = evaluate(agent, config)
            print(f'  >> Eval @ frame {agent.total_frames:,}: '
                  f'score = {eval_mean:.1f} +/- {eval_std:.1f}')
            if eval_mean > best_eval_score:
                best_eval_score = eval_mean
                best_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                agent.save(best_path, extra={'eval_score': eval_mean})

        # Episode end
        if done:
            episode += 1
            episode_rewards.append(ep_reward)

            if episode % config.LOG_FREQ == 0:
                elapsed = time.time() - start_time
                fps     = agent.total_frames / elapsed if elapsed > 0 else 0
                avg_r   = np.mean(episode_rewards)
                avg_l   = np.mean(losses) if losses else 0.0
                beta    = agent.get_beta()
                status  = 'filling' if len(agent.replay_buffer) < config.LEARNING_START \
                          else 'training'
                print(
                    f'Frame: {agent.total_frames:>10,}/{config.TOTAL_FRAMES:,} | '
                    f'Ep: {episode:>5} | '
                    f'Reward: {ep_reward:>7.1f} | '
                    f'Avg(100): {avg_r:>7.1f} | '
                    f'Loss: {avg_l:.4f} | '
                    f'eps: {agent.get_epsilon():.4f} | '
                    f'beta: {beta:.4f} | '
                    f'Buffer: {len(agent.replay_buffer):>7,} | '
                    f'FPS: {fps:>6.1f} | '
                    f'{status}'
                )

            state, _ = env.reset()
            ep_reward = 0.0
            step_in_episode = 0

    agent.save(os.path.join(config.CHECKPOINT_DIR, 'final_model.pth'))
    if action_logger is not None:
        action_logger.close()
    env.close()
    print('Training complete.')


# =============================================================================
# 8. 테스트
# =============================================================================
def test(model_path, config: Config, n_episodes=10, render=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_cfg = Config()
    test_cfg.REWARD_CLIP           = False
    test_cfg.TERMINAL_ON_LIFE_LOSS = False
    env = make_env(test_cfg, render=render)

    agent = D3QNAgent(env, config, device)
    agent.load(model_path)
    agent.q_network.eval()

    # Action Logger 초기화
    action_logger = None
    if config.LOG_ACTIONS:
        action_logger = ActionLogger(base_dir=config.LOG_ACTIONS_DIR, mode="test")

    scores = []
    global_step = 0
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        done  = False
        score = 0.0
        step_in_ep = 0
        while not done:
            action, q_values = agent.select_action(
                state, epsilon=config.EVAL_EPSILON, return_q=True
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            score += reward
            step_in_ep  += 1
            global_step += 1

            # Action 로깅
            if action_logger is not None:
                action_logger.log_step(
                    frame=state,
                    episode=ep,
                    step_in_episode=step_in_ep,
                    global_step=global_step,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    epsilon=config.EVAL_EPSILON,
                    q_values=q_values,
                    loss=None,
                )

            state = next_state
        scores.append(score)
        print(f'  Episode {ep:>2}/{n_episodes}: score = {score:.0f}')

    if action_logger is not None:
        action_logger.close()
    env.close()
    print(f'\nResults over {n_episodes} episodes:')
    print(f'  Mean   : {np.mean(scores):.1f}')
    print(f'  Std    : {np.std(scores):.1f}')
    print(f'  Min    : {np.min(scores):.0f}')
    print(f'  Max    : {np.max(scores):.0f}')
    print(f'  Median : {np.median(scores):.0f}')

    print(f'\nBenchmark (SpaceInvaders, 30 no-ops):')
    print(f'  Random  :    148')
    print(f'  DQN     :  1,976')
    print(f'  DDQN    :  5,909')
    print(f'  D3QN    :  9,015')
    print(f'  D3QN+PER: ~15,000 (expected)')
    print(f'  Ours    : {np.mean(scores):.0f}')


# =============================================================================
# 9. 진입점
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='D3QN+PER v3 (Dueling Double DQN + Prioritized Experience Replay)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python D3QN_v3.py --train
  python D3QN_v3.py --train --resume checkpoints_v3/d3qn_frame_1000000.pth
  python D3QN_v3.py --test  --model  checkpoints_v3/best_model.pth
  python D3QN_v3.py --test  --model  checkpoints_v3/best_model.pth --render --episodes 20
  python D3QN_v3.py --train --life-loss
  python D3QN_v3.py --train --log-actions
  python D3QN_v3.py --test  --model checkpoints_v3/best_model.pth --log-actions --log-dir ./my_logs
        """,
    )
    parser.add_argument('--train',        action='store_true')
    parser.add_argument('--test',         action='store_true')
    parser.add_argument('--model',        type=str,   default=None)
    parser.add_argument('--resume',       type=str,   default=None)
    parser.add_argument('--render',       action='store_true')
    parser.add_argument('--episodes',     type=int,   default=10)
    parser.add_argument('--life-loss',    action='store_true')
    parser.add_argument('--no-life-loss', action='store_true')
    parser.add_argument('--lr',           type=float, default=None)
    parser.add_argument('--total-frames', type=int,   default=None)
    parser.add_argument('--alpha',        type=float, default=None, help='PER alpha (default: 0.6)')
    parser.add_argument('--beta-start',   type=float, default=None, help='PER beta start (default: 0.4)')
    parser.add_argument('--log-actions',  action='store_true',
                        help='매 스텝 action/이미지/Q값 등을 CSV+PNG로 저장')
    parser.add_argument('--log-dir',      type=str, default=None,
                        help='Action 로그 저장 디렉토리 (default: ./action_logs)')

    args   = parser.parse_args()
    config = Config()

    if args.lr:           config.LEARNING_RATE        = args.lr
    if args.total_frames: config.TOTAL_FRAMES          = args.total_frames
    if args.life_loss:    config.TERMINAL_ON_LIFE_LOSS = True
    if args.no_life_loss: config.TERMINAL_ON_LIFE_LOSS = False
    if args.alpha:        config.PER_ALPHA             = args.alpha
    if args.beta_start:   config.PER_BETA_START        = args.beta_start
    if args.log_actions:  config.LOG_ACTIONS           = True
    if args.log_dir:      config.LOG_ACTIONS_DIR       = args.log_dir

    if args.train:
        train(config, resume_path=args.resume)
    elif args.test:
        if not args.model:
            parser.error('--test requires --model MODEL_PATH')
        test(args.model, config, n_episodes=args.episodes, render=args.render)
    else:
        parser.print_help()