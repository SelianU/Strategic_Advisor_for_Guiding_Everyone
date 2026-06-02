"""
ai_agents/d3qn_helper.py
────────────────────────
D3QN (Dueling Double DQN) 공통 모듈.
모든 Atari 게임이 이 하나의 파일을 공유합니다.

사용 예시:
    from ai_agents.d3qn_helper import load_d3qn, get_q_values, GAME_CONFIGS

    net, n_actions = load_d3qn('space_invaders', 'path/to/best_model.pth', device='cuda')
    q = get_q_values(net, stacked_state, device='cuda')
"""

import random
from collections import deque, Counter

import numpy as np
import cv2
import torch
import torch.nn as nn
import gymnasium as gym


# ── 게임별 설정 레지스트리 ─────────────────────────────────────────────────────

GAME_CONFIGS: dict[str, dict] = {
    "space_invaders": {
        "env_name":   "ALE/SpaceInvaders-v5",
        "frame_skip": 4,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",
            1: "FIRE",
            2: "RIGHT",
            3: "LEFT",
            4: "RIGHTFIRE",
            5: "LEFTFIRE",
        },
    },
    "breakout": {
        "env_name":   "ALE/Breakout-v5",
        "frame_skip": 3,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",
            1: "FIRE",
            2: "RIGHT",
            3: "LEFT",
        },
    },
    "enduro": {
        "env_name":   "ALE/Enduro-v5",
        "frame_skip": 4,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",
            1: "FIRE",       # 가속
            2: "RIGHT",
            3: "LEFT",
            4: "DOWN",       # 브레이크
            5: "DOWNRIGHT",
            6: "DOWNLEFT",
            7: "RIGHTFIRE",  # 가속 + 오른쪽
            8: "LEFTFIRE",   # 가속 + 왼쪽
        },
    },
    "alien": {
        "env_name":   "ALE/Alien-v5",
        "frame_skip": 4,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",       1: "FIRE",
            2: "UP",         3: "RIGHT",      4: "LEFT",        5: "DOWN",
            6: "UPRIGHT",    7: "UPLEFT",     8: "DOWNRIGHT",   9: "DOWNLEFT",
            10: "UPFIRE",    11: "RIGHTFIRE", 12: "LEFTFIRE",   13: "DOWNFIRE",
            14: "UPRIGHTFIRE", 15: "UPLEFTFIRE",
            16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE",
        },
    },
    "amidar": {
        "env_name":   "ALE/Amidar-v5",
        "frame_skip": 4,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",  1: "FIRE",      2: "UP",        3: "RIGHT",
            4: "LEFT",  5: "DOWN",      6: "UPFIRE",    7: "RIGHTFIRE",
            8: "LEFTFIRE", 9: "DOWNFIRE",
        },
    },
    "assault": {
        "env_name":   "ALE/Assault-v5",
        "frame_skip": 4,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",  1: "FIRE",  2: "UP",  3: "RIGHT",
            4: "LEFT",  5: "RIGHTFIRE",  6: "LEFTFIRE",
        },
    },
    "asterix": {
        "env_name":   "ALE/Asterix-v5",
        "frame_skip": 4,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",  1: "UP",    2: "RIGHT",     3: "LEFT",
            4: "DOWN",  5: "UPRIGHT", 6: "UPLEFT",  7: "DOWNRIGHT",
            8: "DOWNLEFT",
        },
    },
    "asteroids": {
        "env_name":   "ALE/Asteroids-v5",
        "frame_skip": 4,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",   1: "FIRE",        2: "UP",        3: "RIGHT",
            4: "LEFT",   5: "DOWN",        6: "UPRIGHT",   7: "UPLEFT",
            8: "UPFIRE", 9: "RIGHTFIRE",  10: "LEFTFIRE", 11: "DOWNFIRE",
            12: "UPRIGHTFIRE", 13: "UPLEFTFIRE",
        },
    },
    "atlantis": {
        "env_name":   "ALE/Atlantis-v5",
        "frame_skip": 4,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",  1: "FIRE",  2: "RIGHTFIRE",  3: "LEFTFIRE",
        },
    },
    "mariobros": {
        "env_name":   "ALE/MarioBros-v5",
        "frame_skip": 4,
        "stack_size": 4,
        "img_size":   84,
        "no_op_max":  30,
        "reward_clip": True,
        "terminal_on_life_loss": False,
        "action_names": {
            0: "NOOP",   1: "FIRE",          2: "UP",            3: "RIGHT",
            4: "LEFT",   5: "DOWN",          6: "UPRIGHT",       7: "UPLEFT",
            8: "DOWNRIGHT", 9: "DOWNLEFT",  10: "UPFIRE",       11: "RIGHTFIRE",
            12: "LEFTFIRE", 13: "DOWNFIRE", 14: "UPRIGHTFIRE",  15: "UPLEFTFIRE",
            16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE",
        },
    },
}


# ── Config 헬퍼 ───────────────────────────────────────────────────────────────

class GameConfig:
    """GAME_CONFIGS 딕셔너리를 속성 접근 가능한 객체로 래핑합니다."""

    def __init__(self, game_id: str):
        if game_id not in GAME_CONFIGS:
            raise ValueError(f"알 수 없는 game_id: '{game_id}'. 등록된 게임: {list(GAME_CONFIGS)}")
        cfg = GAME_CONFIGS[game_id]
        self.game_id                = game_id
        self.env_name               = cfg["env_name"]
        self.frame_skip             = cfg["frame_skip"]
        self.stack_size             = cfg["stack_size"]
        self.img_size               = cfg["img_size"]
        self.no_op_max              = cfg["no_op_max"]
        self.reward_clip            = cfg["reward_clip"]
        self.terminal_on_life_loss  = cfg["terminal_on_life_loss"]
        self.action_names: dict[int, str] = cfg["action_names"]


# ── Atari 래퍼 ────────────────────────────────────────────────────────────────

class AtariWrapper(gym.Wrapper):
    """프레임 스킵·그레이스케일 변환·4-프레임 스택을 처리하는 공통 래퍼."""

    def __init__(self, env, cfg: GameConfig):
        super().__init__(env)
        self.cfg      = cfg
        self.frames   = deque(maxlen=cfg.stack_size)
        self._obs_buf = np.zeros((2, 210, 160, 3), dtype=np.uint8)
        self.lives    = 0

        low  = np.zeros((cfg.stack_size, cfg.img_size, cfg.img_size), dtype=np.uint8)
        high = np.full_like(low, 255)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.cfg.img_size, self.cfg.img_size),
                          interpolation=cv2.INTER_AREA)

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        for _ in range(random.randint(1, self.cfg.no_op_max)):
            frame, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                frame, info = self.env.reset(**kwargs)
        proc = self._preprocess(frame)
        for _ in range(self.cfg.stack_size):
            self.frames.append(proc)
        self.lives = self.env.unwrapped.ale.lives()
        return self._obs(), info

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self.cfg.frame_skip):
            frame, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if i == self.cfg.frame_skip - 2:
                self._obs_buf[0] = frame
            if i == self.cfg.frame_skip - 1:
                self._obs_buf[1] = frame
            if terminated or truncated:
                break

        self.frames.append(self._preprocess(self._obs_buf.max(axis=0)))

        if self.cfg.terminal_on_life_loss:
            cur = self.env.unwrapped.ale.lives()
            if cur < self.lives:
                terminated = True
            self.lives = cur

        clipped = float(np.clip(total_reward, -1., 1.)) if self.cfg.reward_clip \
                  else total_reward
        return self._obs(), clipped, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        return np.array(self.frames, dtype=np.uint8)


# ── 신경망 ────────────────────────────────────────────────────────────────────

class GradRescale(nn.Module):
    SCALE = 1.0 / (2.0 ** 0.5)

    def forward(self, x):
        return x

    def backward_hook(self, module, grad_input, grad_output):
        return tuple(g * self.SCALE if g is not None else None for g in grad_input)


class DuelingDQN(nn.Module):
    """Dueling Double DQN 네트워크 (모든 게임 공통)."""

    def __init__(self, input_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(c,  32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, *input_shape)).flatten().shape[0]

        self.grad_rescale = GradRescale()
        self.grad_rescale.register_full_backward_hook(self.grad_rescale.backward_hook)

        self.value_stream     = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, n_actions))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x         = x.contiguous().float().mul_(1.0 / 255.0)
        features  = self.features(x).flatten(1)
        features  = self.grad_rescale(features)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ── 공개 API ──────────────────────────────────────────────────────────────────

def load_d3qn(game_id: str, model_path: str, device: str = 'cpu') -> tuple[DuelingDQN, int]:
    """
    지정한 게임의 D3QN 모델을 로드합니다.

    Args:
        game_id:    GAME_CONFIGS에 등록된 게임 식별자 ('space_invaders', 'breakout', 'enduro' …)
        model_path: 체크포인트 파일 경로 (torch.save 포맷, 'q_network' 키 포함)
        device:     'cpu' 또는 'cuda'

    Returns:
        (net, n_actions) 튜플
    """
    cfg = GameConfig(game_id)
    env = gym.make(cfg.env_name, frameskip=1)
    n_actions = env.action_space.n
    env.close()

    input_shape = (cfg.stack_size, cfg.img_size, cfg.img_size)
    net = DuelingDQN(input_shape, n_actions).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['q_network']
    # torch.compile 흔적 제거
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    net.load_state_dict(state_dict)
    net.eval()
    print(f"✅ [{game_id}] D3QN 로드: {model_path}  (frame={ckpt.get('frame', '?'):,})")
    return net, n_actions


def get_q_values(net: DuelingDQN, state: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """4-프레임 스택 상태에서 Q-value 배열을 반환합니다."""
    s = torch.as_tensor(state, dtype=torch.uint8, device=device).unsqueeze(0)
    with torch.no_grad():
        q = net(s).squeeze(0).cpu().numpy()
    return q


def analyze_episode(
    game_id: str,
    net: DuelingDQN,
    episode_data: list[dict],
    device: str = 'cpu',
) -> dict:
    """
    에피소드 전체 스텝을 분석해 Q-value 손실·일치율·최악 행동을 반환합니다.

    Args:
        game_id:      액션 이름 매핑에 사용할 게임 식별자
        net:          로드된 DuelingDQN 모델
        episode_data: 각 스텝의 {'stacked_state', 'action', 'reward'} 딕셔너리 목록
        device:       'cpu' 또는 'cuda'

    Returns:
        {analyses, worst, avg_loss, total_steps, action_counts, ai_action_counts, agreement_rate}
    """
    action_names = GAME_CONFIGS[game_id]["action_names"]
    analyses = []

    for i, entry in enumerate(episode_data):
        state  = entry.get('stacked_state')
        action = entry.get('action')
        if state is None or action is None:
            continue

        q_vals   = get_q_values(net, state, device)
        best_act = int(np.argmax(q_vals))
        player_q = float(q_vals[action])
        best_q   = float(q_vals[best_act])

        analyses.append({
            'step':             i,
            'action':           int(action),
            'action_name':      action_names.get(action, str(action)),
            'best_action':      best_act,
            'best_action_name': action_names.get(best_act, str(best_act)),
            'q_values':         [round(float(v), 3) for v in q_vals],
            'player_q':         round(player_q, 3),
            'best_q':           round(best_q, 3),
            'loss':             round(best_q - player_q, 3),
            'reward':           float(entry.get('reward', 0)),
            'is_best':          (action == best_act),
        })

    if not analyses:
        return {'analyses': [], 'worst': None, 'avg_loss': 0}

    losses   = [a['loss'] for a in analyses]
    avg_loss = round(float(np.mean(losses)), 3)
    worst    = max(analyses, key=lambda a: a['loss'])

    action_counts = Counter(a['action'] for a in analyses)
    best_counts   = Counter(a['best_action'] for a in analyses)

    return {
        'analyses':         analyses,
        'worst':            worst,
        'avg_loss':         avg_loss,
        'total_steps':      len(analyses),
        'action_counts':    {action_names.get(k, str(k)): v for k, v in action_counts.items()},
        'ai_action_counts': {action_names.get(k, str(k)): v for k, v in best_counts.items()},
        'agreement_rate':   round(
            sum(1 for a in analyses if a['is_best']) / len(analyses) * 100, 1
        ),
    }