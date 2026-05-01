import random
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn as nn
import gymnasium as gym


ACTION_NAMES = {
    0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT'
}


class Config:
    ENV_NAME   = "ALE/Breakout-v5"
    FRAME_SKIP = 3
    STACK_SIZE = 4
    IMG_SIZE   = 84
    NO_OP_MAX  = 30
    REWARD_CLIP    = True
    TERMINAL_ON_LIFE_LOSS = False


class AtariWrapper(gym.Wrapper):
    def __init__(self, env, config: Config):
        super().__init__(env)
        self.cfg    = config
        self.frames = deque(maxlen=config.STACK_SIZE)
        self._obs_buf = np.zeros((2, 210, 160, 3), dtype=np.uint8)
        self.lives  = 0

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
            cur = self.env.unwrapped.ale.lives()
            if cur < self.lives:
                terminated = True
            self.lives = cur

        clipped = float(np.clip(total_reward, -1., 1.)) if self.cfg.REWARD_CLIP \
                  else total_reward
        return self._obs(), clipped, terminated, truncated, info

    def _obs(self):
        return np.array(self.frames, dtype=np.uint8)


class GradRescale(nn.Module):
    SCALE = 1.0 / (2.0 ** 0.5)
    def forward(self, x):
        return x
    def backward_hook(self, module, grad_input, grad_output):
        return tuple(g * self.SCALE if g is not None else None for g in grad_input)


class DuelingDQN(nn.Module):
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

        self.grad_rescale = GradRescale()
        self.grad_rescale.register_full_backward_hook(self.grad_rescale.backward_hook)

        self.value_stream     = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, n_actions))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x         = x.contiguous().float().mul_(1.0 / 255.0)
        features  = self.features(x).flatten(1)
        features  = self.grad_rescale(features)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def load_breakout_d3qn(model_path: str, device='cpu'):
    cfg = Config()
    n_actions = gym.make(cfg.ENV_NAME, frameskip=1).action_space.n
    input_shape = (cfg.STACK_SIZE, cfg.IMG_SIZE, cfg.IMG_SIZE)

    net = DuelingDQN(input_shape, n_actions).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = ckpt['q_network']
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    net.load_state_dict(state_dict)
    net.eval()
    print(f"✅ Breakout D3QN 로드: {model_path}  (frame={ckpt.get('frame', '?'):,})")
    return net, n_actions


def get_q_values(net, state: np.ndarray, device='cpu') -> np.ndarray:
    s = torch.as_tensor(state, dtype=torch.uint8, device=device).unsqueeze(0)
    with torch.no_grad():
        q = net(s).squeeze(0).cpu().numpy()
    return q


def analyze_episode(net, episode_data: list, device='cpu') -> dict:
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
        loss     = best_q - player_q

        analyses.append({
            'step':       i,
            'action':     int(action),
            'action_name': ACTION_NAMES.get(action, str(action)),
            'best_action': best_act,
            'best_action_name': ACTION_NAMES.get(best_act, str(best_act)),
            'q_values':   [round(float(v), 3) for v in q_vals],
            'player_q':   round(player_q, 3),
            'best_q':     round(best_q, 3),
            'loss':       round(loss, 3),
            'reward':     float(entry.get('reward', 0)),
            'is_best':    (action == best_act),
        })

    if not analyses:
        return {'analyses': [], 'worst': None, 'avg_loss': 0}

    losses   = [a['loss'] for a in analyses]
    avg_loss = round(float(np.mean(losses)), 3)
    worst    = max(analyses, key=lambda a: a['loss'])

    from collections import Counter
    action_counts = Counter(a['action'] for a in analyses)
    best_counts   = Counter(a['best_action'] for a in analyses)

    return {
        'analyses':      analyses,
        'worst':         worst,
        'avg_loss':      avg_loss,
        'total_steps':   len(analyses),
        'action_counts': {ACTION_NAMES.get(k, str(k)): v for k, v in action_counts.items()},
        'ai_action_counts': {ACTION_NAMES.get(k, str(k)): v for k, v in best_counts.items()},
        'agreement_rate': round(sum(1 for a in analyses if a['is_best']) / len(analyses) * 100, 1),
    }
