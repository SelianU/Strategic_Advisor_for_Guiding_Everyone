import random
import numpy as np


class SumTree:
    """PER의 핵심 자료구조. O(log n) 샘플링 / 업데이트. [Schaul et al. 2015]"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.pos      = 0
        self.size     = 0

    def _propagate(self, idx: int, delta: float):
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority: float) -> int:
        leaf_idx = self.pos + self.capacity - 1
        self.update(leaf_idx, priority)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return leaf_idx

    def get(self, value: float) -> tuple:
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
    def total(self) -> float:
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        return self.tree[self.capacity - 1:self.capacity - 1 + self.size].max()


class PrioritizedReplayBuffer:
    """SumTree 기반 Prioritized Experience Replay 버퍼."""

    def __init__(self, capacity: int, state_shape: tuple, alpha: float):
        self.capacity    = capacity
        self.alpha       = alpha
        self.state_shape = state_shape

        self.tree = SumTree(capacity)

        # next_state는 states[idx+1]로 참조 → capacity+1 크기로 메모리 절약
        self.states  = np.zeros((capacity + 1, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.float32)

        self._pos = 0

    def push(self, state, action, reward, next_state, done):
        max_p    = self.tree.max_priority if self.tree.size > 0 else 1.0
        priority = max_p ** self.alpha

        self.states[self._pos]     = state
        self.states[self._pos + 1] = next_state
        self.actions[self._pos]    = action
        self.rewards[self._pos]    = reward
        self.dones[self._pos]      = float(done)

        self.tree.add(priority)
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> tuple:
        total   = self.tree.total
        segment = total / batch_size

        leaf_idxs, data_idxs, priorities = [], [], []
        for i in range(batch_size):
            v = random.uniform(segment * i, segment * (i + 1))
            leaf_idx, p, data_idx = self.tree.get(v)
            leaf_idxs.append(leaf_idx)
            data_idxs.append(data_idx)
            priorities.append(p)

        data_idxs  = np.array(data_idxs)
        next_idxs  = data_idxs + 1
        priorities = np.array(priorities, dtype=np.float64)

        N          = self.tree.size
        probs      = priorities / (total + 1e-8)
        is_weights = (N * probs) ** (-beta)
        is_weights = (is_weights / is_weights.max()).astype(np.float32)

        return (
            self.states[data_idxs],
            self.actions[data_idxs],
            self.rewards[data_idxs],
            self.states[next_idxs],
            self.dones[data_idxs],
            leaf_idxs,
            is_weights,
        )

    def update_priorities(self, leaf_idxs, td_errors, eps: float):
        for idx, err in zip(leaf_idxs, td_errors):
            p = (abs(err) + eps) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self) -> int:
        return self.tree.size
