"""
teacher_agent/tools/reward.py
──────────────────────────────
Teacher Agent 전용 reward 함수.

Atari 게임 reward와 독립적으로 정의됨.
현재는 placeholder — 추후 실제 로직으로 채울 것.

사용체: trainer.py 의 train() 루프 내
    teacher_reward = compute_teacher_reward(...)
"""
from __future__ import annotations
from typing import Any


def compute_teacher_reward(
    obs:        Any   = None,
    lives:      int   = None,
    action:     float = None,
    intervened: bool  = None,
    next_obs:   Any   = None,
    next_lives: int   = None,
    info:       dict  = None,
    **kwargs,
) -> float:
    """
    Teacher Agent reward 계산.

    Args:
        obs        : 개입 결정 시점의 observation
        lives      : 개입 결정 시점의 남은 목숨
        action     : teacher가 출력한 개입 확률 ∈ [0, 1]
        intervened : threshold 기준 실제 개입 여부
        next_obs   : 다음 스텝 observation
        next_lives : 다음 스텝 남은 목숨
        info       : env.step() 반환 info dict

    Returns:
        float : teacher reward

    TODO:
        - 실제 reward 설계 로직 구현
    """
    # ────────────────────────────────────────────────────────────────────────
    # TODO: 여기에 teacher reward 로직 구현
    # ────────────────────────────────────────────────────────────────────────
    raise NotImplementedError("compute_teacher_reward() 는 아직 구현되지 않았습니다.")
