import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import Config
from .model import DuelingDQN
from .buffer import PrioritizedReplayBuffer


class D3QNAgent:
    """Dueling Double DQN + PER 에이전트."""

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

        if hasattr(torch, 'compile'):
            try:
                self.q_network      = torch.compile(self.q_network,      backend='eager')
                self.target_network = torch.compile(self.target_network, backend='eager')
                print('  [torch.compile] applied (backend=eager)')
            except Exception as e:
                print(f'  [torch.compile] skipped: {e}')

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

    # ------------------------------------------------------------------
    # 스케줄
    # ------------------------------------------------------------------

    def get_epsilon(self) -> float:
        frac = min(1.0, self.total_frames / self.cfg.EPSILON_DECAY)
        return self.cfg.EPSILON_START - frac * (self.cfg.EPSILON_START - self.cfg.EPSILON_END)

    def get_beta(self) -> float:
        frac = min(1.0, self.total_frames / self.cfg.PER_BETA_FRAMES)
        return self.cfg.PER_BETA_START + frac * (1.0 - self.cfg.PER_BETA_START)

    # ------------------------------------------------------------------
    # 행동 선택
    # ------------------------------------------------------------------

    def select_action(self, state, epsilon: float = None) -> int:
        eps = epsilon if epsilon is not None else self.get_epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)
        s = torch.as_tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q_network(s).argmax().item()

    # ------------------------------------------------------------------
    # 학습 스텝
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
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
            next_a   = self.q_network(ns).argmax(1, keepdim=True)
            next_q   = self.target_network(ns).gather(1, next_a).squeeze(1)
            target_q = r + self.cfg.DISCOUNT_FACTOR * next_q * (1.0 - d)

        td_errors        = (target_q - current_q).detach().cpu().numpy()
        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss             = (w * elementwise_loss).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.cfg.GRAD_CLIP_NORM)
        self.optimizer.step()

        self.replay_buffer.update_priorities(leaf_idxs, td_errors, self.cfg.PER_EPS)
        return loss.item()

    # ------------------------------------------------------------------
    # 타깃 네트워크 업데이트
    # ------------------------------------------------------------------

    def update_target(self, soft: bool = False, tau: float = 0.005):
        if soft:
            for t_p, q_p in zip(self.target_network.parameters(),
                                  self.q_network.parameters()):
                t_p.data.copy_(tau * q_p.data + (1.0 - tau) * t_p.data)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())

    # ------------------------------------------------------------------
    # 체크포인트
    # ------------------------------------------------------------------

    def save(self, path: str, extra: dict = None):
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

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(ckpt['q_network'])
        self.target_network.load_state_dict(ckpt.get('target', ckpt['q_network']))
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_frames = ckpt.get('frame', 0)
        print(f'  [Loaded] {path}  (frame={self.total_frames:,})')
