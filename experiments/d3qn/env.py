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
      - max-pooling on last 2 frames (flickering 제거) [1]
      - random no-op reset [1]
      - grayscale + 84x84 resize [1]
      - frame stacking (4 frames) [1]
      - optional terminal_on_life_loss
      - eval 모드에서 목숨 손실 시 frame stack 리셋 (학습 분포와 일치)
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

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
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

        cur_lives = self.env.unwrapped.ale.lives()
        if self.cfg.TERMINAL_ON_LIFE_LOSS:
            if cur_lives < self.lives:
                terminated = True
        else:
            # eval 모드: 목숨 손실 시 frame stack을 현재 프레임으로 채워
            # 학습 시 reset()이 frame stack을 초기화하는 것과 분포를 맞춤
            if cur_lives < self.lives:
                proc = self._preprocess(self._obs_buf.max(axis=0))
                for _ in range(self.cfg.STACK_SIZE):
                    self.frames.append(proc)
        self.lives = cur_lives

        clipped_reward = (float(np.clip(total_reward, -1.0, 1.0))
                          if self.cfg.REWARD_CLIP else total_reward)
        info['raw_reward'] = total_reward
        return self._obs(), clipped_reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        return np.array(self.frames, dtype=np.uint8)


class RenderWrapper(gym.Wrapper):
    """
    AtariWrapper 내부의 프레임 루프를 직접 구동해 매 ALE 프레임을 60fps로 렌더링.
    에이전트는 FRAME_SKIP마다 한 번 행동을 선택하지만 화면은 매 프레임 갱신된다.
    """

    def __init__(self, env, scale: float = 2.0, fps: int = 60,
                 record_path: str = None, show: bool = True):
        super().__init__(env)
        self.scale       = scale
        self.fps         = fps
        self.record_path = record_path
        self.show        = show   # False → 창 없이 녹화만
        self._window     = None
        self._clock      = None
        self._writer     = None  # cv2.VideoWriter

    def _show_frame(self, frame: np.ndarray):
        """raw RGB 프레임(H×W×3)을 처리. 녹화 중이면 파일에 기록, show=True면 pygame 창에도 표시."""
        h, w   = frame.shape[:2]
        nw, nh = int(w * self.scale), int(h * self.scale)

        # cv2로 리사이즈 (pygame과 영상 모두 동일 배열 사용)
        scaled = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # 영상 녹화
        if self.record_path is not None:
            if self._writer is None:
                fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
                self._writer = cv2.VideoWriter(self.record_path, fourcc, self.fps, (nw, nh))
            self._writer.write(cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR))

        # pygame 화면 출력 (show=True일 때만)
        if not self.show:
            return
        import pygame
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((nw, nh))
            pygame.display.set_caption("D3QN — SpaceInvaders")
            self._clock  = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(scaled.transpose(1, 0, 2))
        self._window.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self._clock.tick(self.fps)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self.env.env.render()
        if frame is not None:
            self._show_frame(frame)
        return obs, info

    def step(self, action):
        """
        AtariWrapper.step()을 우회해 ALE 프레임을 한 장씩 직접 구동.
        AtariWrapper의 max-pool / preprocess / life-loss 로직은 그대로 재현.
        """
        cfg       = self.env.cfg
        inner_env = self.env.env  # raw ALE gym env

        total_reward = 0.0
        terminated = truncated = False

        for i in range(cfg.FRAME_SKIP):
            frame, reward, terminated, truncated, info = inner_env.step(action)
            total_reward += reward
            if i == cfg.FRAME_SKIP - 2:
                self.env._obs_buf[0] = frame
            if i == cfg.FRAME_SKIP - 1:
                self.env._obs_buf[1] = frame
            self._show_frame(frame)  # 매 프레임 렌더
            if terminated or truncated:
                break

        self.env.frames.append(self.env._preprocess(self.env._obs_buf.max(axis=0)))

        cur_lives = inner_env.unwrapped.ale.lives()
        if cfg.TERMINAL_ON_LIFE_LOSS:
            if cur_lives < self.env.lives:
                terminated = True
        else:
            if cur_lives < self.env.lives:
                proc = self.env._preprocess(self.env._obs_buf.max(axis=0))
                for _ in range(cfg.STACK_SIZE):
                    self.env.frames.append(proc)
        self.env.lives = cur_lives

        clipped_reward = (float(np.clip(total_reward, -1.0, 1.0))
                          if cfg.REWARD_CLIP else total_reward)
        return self.env._obs(), clipped_reward, terminated, truncated, info

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f'  [Video saved] {self.record_path}')
        if self._window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._window = None
        super().close()


def make_env(config: Config, render: bool = False, render_scale: float = 2.0,
             fps: int = 60, record_path: str = None):
    need_frames = render or bool(record_path)
    env = gym.make(
        config.ENV_NAME,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="rgb_array" if need_frames else None,
    )
    wrapped = AtariWrapper(env, config)
    if need_frames:
        wrapped = RenderWrapper(wrapped, scale=render_scale, fps=fps,
                                record_path=record_path, show=render)
    return wrapped
