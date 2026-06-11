"""
teacher_agent/tools/agent.py
──────────────────────────────
Teacher SAC Agent (Continuous, 1-D action).

State  = (obs: 4×84×84) + (lives: scalar)
Action = intervention_prob ∈ [0, 1]
Intervene = prob >= THRESHOLD
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config  import Config
from .buffer  import ReplayBuffer
from .network import TeacherActor, TeacherCritic


class TeacherSACAgent:
    """
    Teacher SAC Agent.

    주요 메서드:
        select_action(obs, lives)        → (prob, intervene)
        get_q_values(obs, lives, action) → Q-value numpy
        train_step(step)                 → (qf_loss, actor_loss, alpha_loss)
        save(path) / load(path)
    """

    def __init__(self, env, config: Config, device: torch.device):
        self.cfg          = config
        self.device       = device
        self.input_shape  = env.observation_space.shape
        self.total_frames = 0

        # ── 네트워크 ──────────────────────────────────────────────────────────
        self.actor   = TeacherActor(self.input_shape, config.MAX_LIVES).to(device)
        self.qf1     = TeacherCritic(self.input_shape, config.MAX_LIVES).to(device)
        self.qf2     = TeacherCritic(self.input_shape, config.MAX_LIVES).to(device)
        self.qf1_tgt = TeacherCritic(self.input_shape, config.MAX_LIVES).to(device)
        self.qf2_tgt = TeacherCritic(self.input_shape, config.MAX_LIVES).to(device)
        self.qf1_tgt.load_state_dict(self.qf1.state_dict())
        self.qf2_tgt.load_state_dict(self.qf2.state_dict())

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=config.Q_LR, eps=config.ADAM_EPS,
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.POLICY_LR, eps=config.ADAM_EPS,
        )

        # ── Automatic entropy tuning ───────────────────────────────────────────
        self.target_entropy = config.TARGET_ENTROPY
        if config.AUTOTUNE:
            self.log_alpha   = torch.zeros(1, requires_grad=True, device=device)
            self.alpha       = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam(
                [self.log_alpha], lr=config.Q_LR, eps=config.ADAM_EPS,
            )
        else:
            self.alpha = config.ALPHA

        # ── Replay Buffer ──────────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(config.REPLAY_CAPACITY, self.input_shape)

    # ── 추론 ──────────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, lives: int):
        """
        개입 여부 결정.

        Args:
            obs   : (STACK, H, W) uint8
            lives : 현재 남은 목숨 수

        Returns:
            prob      (float) : 개입 확률 ∈ [0, 1]
            intervene (bool)  : prob >= THRESHOLD
        """
        s = torch.as_tensor(obs,   dtype=torch.uint8, device=self.device).unsqueeze(0)
        l = torch.tensor([lives],  dtype=torch.int32, device=self.device)
        with torch.no_grad():
            prob, _, intervene = self.actor.get_action(s, l, self.cfg.THRESHOLD)
        return prob.item(), intervene.item()

    def get_q_values(self, obs: np.ndarray, lives: int, action: float) -> float:
        """현재 (state, action)에 대한 Q-value (두 critic 평균)."""
        s = torch.as_tensor(obs,      dtype=torch.uint8,   device=self.device).unsqueeze(0)
        l = torch.tensor([lives],     dtype=torch.int32,   device=self.device)
        a = torch.tensor([action],    dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = (self.qf1(s, l, a) + self.qf2(s, l, a)) / 2.0
        return q.item()

    # ── 학습 ──────────────────────────────────────────────────────────────────

    def train_step(self, step: int):
        """
        SAC 업데이트 한 스텝 (Continuous 1-D action).

        Returns:
            (qf_loss, actor_loss, alpha_loss) 또는 버퍼 미달 시 None.
        """
        if len(self.replay_buffer) < self.cfg.LEARNING_START:
            return None

        states, lives, actions, rewards, next_states, next_lives, dones =             self.replay_buffer.sample(self.cfg.BATCH_SIZE)

        s   = torch.as_tensor(states,      dtype=torch.uint8,   device=self.device)
        l   = torch.as_tensor(lives,       dtype=torch.int32,   device=self.device)
        a   = torch.as_tensor(actions,     dtype=torch.float32, device=self.device)
        r   = torch.as_tensor(rewards,     dtype=torch.float32, device=self.device)
        ns  = torch.as_tensor(next_states, dtype=torch.uint8,   device=self.device)
        nl  = torch.as_tensor(next_lives,  dtype=torch.int32,   device=self.device)
        d   = torch.as_tensor(dones,       dtype=torch.float32, device=self.device)

        # ── Critic 업데이트 ────────────────────────────────────────────────────
        with torch.no_grad():
            next_prob, next_log_prob, _ = self.actor.get_action(
                ns, nl, self.cfg.THRESHOLD,
            )
            q1_next = self.qf1_tgt(ns, nl, next_prob).squeeze(-1)
            q2_next = self.qf2_tgt(ns, nl, next_prob).squeeze(-1)
            next_q  = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = r + (1.0 - d) * self.cfg.DISCOUNT_FACTOR * next_q

        qf1_loss = F.mse_loss(self.qf1(s, l, a).squeeze(-1), target_q)
        qf2_loss = F.mse_loss(self.qf2(s, l, a).squeeze(-1), target_q)
        qf_loss  = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.q_optimizer.step()

        # ── Actor 업데이트 ─────────────────────────────────────────────────────
        prob, log_prob, _ = self.actor.get_action(s, l, self.cfg.THRESHOLD)
        with torch.no_grad():
            q1 = self.qf1(s, l, prob).squeeze(-1)
            q2 = self.qf2(s, l, prob).squeeze(-1)
            min_q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Alpha 업데이트 ─────────────────────────────────────────────────────
        alpha_loss = None
        if self.cfg.AUTOTUNE:
            alpha_loss = (
                -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
            ).mean()
            self.a_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ── Target Network Hard Update ─────────────────────────────────────────
        if step % self.cfg.TARGET_UPDATE_FREQ == 0:
            tau = self.cfg.TAU
            for p, tp in zip(self.qf1.parameters(), self.qf1_tgt.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
            for p, tp in zip(self.qf2.parameters(), self.qf2_tgt.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        return (
            qf_loss.item(),
            actor_loss.item(),
            alpha_loss.item() if alpha_loss is not None else None,
        )

    # ── 저장 / 로드 ────────────────────────────────────────────────────────────

    def save(self, path: str, extra: dict = None):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        ckpt = {
            "frame":         self.total_frames,
            "actor":         self.actor.state_dict(),
            "qf1":           self.qf1.state_dict(),
            "qf2":           self.qf2.state_dict(),
            "qf1_tgt":       self.qf1_tgt.state_dict(),
            "qf2_tgt":       self.qf2_tgt.state_dict(),
            "q_optimizer":   self.q_optimizer.state_dict(),
            "act_optimizer": self.actor_optimizer.state_dict(),
            "alpha":         self.alpha,
        }
        if self.cfg.AUTOTUNE:
            ckpt["log_alpha"]   = self.log_alpha.data
            ckpt["a_optimizer"] = self.a_optimizer.state_dict()
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f"  [Saved] {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.qf1.load_state_dict(ckpt["qf1"])
        self.qf2.load_state_dict(ckpt["qf2"])
        self.qf1_tgt.load_state_dict(ckpt["qf1_tgt"])
        self.qf2_tgt.load_state_dict(ckpt["qf2_tgt"])
        self.q_optimizer.load_state_dict(ckpt["q_optimizer"])
        self.actor_optimizer.load_state_dict(ckpt["act_optimizer"])
        self.alpha        = ckpt.get("alpha", self.cfg.ALPHA)
        self.total_frames = ckpt.get("frame", 0)
        if self.cfg.AUTOTUNE and "log_alpha" in ckpt:
            self.log_alpha.data = ckpt["log_alpha"]
            self.a_optimizer.load_state_dict(ckpt["a_optimizer"])
        print(f"  [Loaded] {path}  (frame={self.total_frames:,})")
