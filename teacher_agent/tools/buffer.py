"""
teacher_agent/tools/buffer.py
──────────────────────────────
Teacher Agent 용 Replay Buffer.

기존 대비 변경:
  - lives / next_lives 배열 추가 (states 배열과 동일한 +1 슬롯 트릭 사용)
  - sample() 반환값에 lives, next_lives 포함
"""
import numpy as np


class ReplayBuffer:
    """
    Teacher Agent Replay Buffer.

    저장 구조:
        states[i]      : obs at step i        (uint8, STACK×H×W)
        states[i+1]    : obs at step i+1      (next_state 참조용)
        lives_arr[i]   : lives at step i      (int32)
        lives_arr[i+1] : lives at step i+1    (next_lives 참조용)
        actions[i]     : intervention prob     (float32, scalar)
        rewards[i]     : teacher reward        (float32)
        dones[i]       : episode done flag     (float32)
    """

    def __init__(self, capacity: int, state_shape: tuple):
        self.capacity    = capacity
        self.state_shape = state_shape

        self.states    = np.zeros((capacity + 1, *state_shape), dtype=np.uint8)
        self.lives_arr = np.zeros(capacity + 1,                 dtype=np.int32)
        self.actions   = np.zeros(capacity,                     dtype=np.float32)
        self.rewards   = np.zeros(capacity,                     dtype=np.float32)
        self.dones     = np.zeros(capacity,                     dtype=np.float32)

        self._pos  = 0
        self._size = 0

    def push(
        self,
        state:      np.ndarray,
        lives:      int,
        action:     float,
        reward:     float,
        next_state: np.ndarray,
        next_lives: int,
        done:       bool,
    ):
        self.states[self._pos]     = state
        self.states[self._pos + 1] = next_state
        self.lives_arr[self._pos]     = lives
        self.lives_arr[self._pos + 1] = next_lives
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.dones[self._pos]   = float(done)

        self._pos  = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Returns:
            states, lives, actions, rewards, next_states, next_lives, dones
        """
        idxs      = np.random.randint(0, self._size, size=batch_size)
        next_idxs = idxs + 1
        return (
            self.states[idxs],
            self.lives_arr[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.states[next_idxs],
            self.lives_arr[next_idxs],
            self.dones[idxs],
        )

    def __len__(self) -> int:
        return self._size
