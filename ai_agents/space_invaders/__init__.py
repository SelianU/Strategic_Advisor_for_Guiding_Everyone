"""
ai_agents/space_invaders/__init__.py
─────────────────────────────────────
하위 호환 shim — 기존 import 경로를 유지하면서 공통 d3qn_helper로 위임합니다.

기존 코드:
    from ai_agents.space_invaders import load_d3qn, get_q_values

위 import는 변경 없이 계속 동작합니다.
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

_GAME_ID     = "space_invaders"
ACTION_NAMES = GAME_CONFIGS[_GAME_ID]["action_names"]


def load_d3qn(model_path: str, device: str = 'cpu'):
    """Space Invaders D3QN 모델을 로드합니다."""
    return _load(_GAME_ID, model_path, device)


__all__ = [
    "ACTION_NAMES",
    "DuelingDQN",
    "AtariWrapper",
    "GradRescale",
    "load_d3qn",
    "get_q_values",
    "analyze_episode",
]