"""
ai_agents/asteroids/__init__.py
──────────────────────────────
하위 호환 shim — 공통 d3qn_helper로 위임합니다.
"""

from ai_agents.d3qn_helper import (
    DuelingDQN,
    AtariWrapper,
    GradRescale,
    get_q_values,
    analyze_episode,
    GAME_CONFIGS,
    load_d3qn as _load,
)

_GAME_ID     = "asteroids"
ACTION_NAMES = GAME_CONFIGS[_GAME_ID]["action_names"]


def load_asteroids_d3qn(model_path: str, device: str = 'cpu'):
    """asteroids D3QN 모델을 로드합니다."""
    return _load(_GAME_ID, model_path, device)


__all__ = [
    "ACTION_NAMES",
    "DuelingDQN",
    "AtariWrapper",
    "GradRescale",
    "load_asteroids_d3qn",
    "get_q_values",
    "analyze_episode",
]
