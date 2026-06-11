"""
teacher_agent/tools/env.py
───────────────────────────
Atari 환경 래퍼.

변경 사항:
  - reset() / step() 의 info dict에 'lives' 키 추가.
    → agent가 info['lives'] 로 현재 남은 목숨을 얻을 수 있음.
"""
import random
from collections import deque

import numpy as np
import cv2

import gymnasium as gym
import ale_py  # noqa: F401

from .config import Config


class AtariWrapper(gym.Wrapper):
    """
    Atari 전처리 래퍼.
    - max-pooling on last 2 frames (flickering 제거)
    - random no-op reset
    - grayscale + 84×84 resize
    - frame stacking (STACK_SIZE frames)
    - optional terminal_on_life_loss

    info dict:
        info['lives'] : 현재 스텝 이후 남은 목숨 수 (int)
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

    # ── 전처리 ────────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.cfg.IMG_SIZE, self.cfg.IMG_SIZE),
                          interpolation=cv2.INTER_AREA)

    def _obs(self) -> np.ndarray:
        return np.array(self.frames, dtype=np.uint8)

    # ── reset ─────────────────────────────────────────────────────────────────

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
        info["lives"] = self.lives          # ← lives 추가
        return self._obs(), info

    # ── step ──────────────────────────────────────────────────────────────────

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

        clipped_reward = (
            float(np.clip(total_reward, -1.0, 1.0)) if self.cfg.REWARD_CLIP
            else float(total_reward)
        )
        info["lives"] = self.lives          # ← lives 추가
        return self._obs(), clipped_reward, terminated, truncated, info


def make_env(config: Config, render: bool = False) -> AtariWrapper:
    """AtariWrapper로 감싼 환경 생성."""
    env = gym.make(
        config.ENV_NAME,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="human" if render else None,
    )
    return AtariWrapper(env, config)
