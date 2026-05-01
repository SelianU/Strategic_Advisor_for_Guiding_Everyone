"""
ai_agents 패키지
────────────────
모든 Atari D3QN 게임이 공유하는 단일 진입점.

사용 예시:
    from ai_agents import load_d3qn, get_q_values, GAME_CONFIGS, ACTION_NAMES

    net, n_actions = load_d3qn('space_invaders', 'path/to/best_model.pth', device='cuda')
    q = get_q_values(net, stacked_state, device='cuda')
    names = ACTION_NAMES('breakout')   # {0: 'NOOP', 1: 'FIRE', ...}
"""

from ai_agents.d3qn_helper import (
    GAME_CONFIGS,
    GameConfig,
    AtariWrapper,
    GradRescale,
    DuelingDQN,
    load_d3qn,
    get_q_values,
    analyze_episode,
)


def ACTION_NAMES(game_id: str) -> dict[int, str]:
    """게임 ID에 해당하는 액션 이름 딕셔너리를 반환합니다."""
    return GAME_CONFIGS[game_id]["action_names"]


__all__ = [
    "GAME_CONFIGS",
    "ACTION_NAMES",
    "GameConfig",
    "AtariWrapper",
    "GradRescale",
    "DuelingDQN",
    "load_d3qn",
    "get_q_values",
    "analyze_episode",
]