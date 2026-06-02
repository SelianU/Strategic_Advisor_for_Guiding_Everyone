"""games/atari/ach_helper.py — 도전과제 판정 공통 헬퍼"""
from __future__ import annotations


def episode_score(data: list) -> float:
    """에피소드 전체 보상 합계."""
    return sum(d.get('reward', 0) for d in data)


def combo_max(data: list, window: int = 200,
              min_r: float = 0, max_r: float = 1e9) -> int:
    """슬라이딩 윈도우 내 최대 킬 수.

    window 스텝 안에서 min_r < reward <= max_r 인 스텝 수의 최댓값을 반환.
    """
    kill_steps = [
        d['step'] for d in data
        if min_r < d.get('reward', 0) <= max_r
    ]
    if not kill_steps:
        return 0
    best, li = 0, 0
    for ri, s in enumerate(kill_steps):
        while kill_steps[li] < s - window:
            li += 1
        best = max(best, ri - li + 1)
    return best


def streak_max(data: list, gap: int = 60,
               min_r: float = 0, max_r: float = 1e9) -> int:
    """연속 킬 최대 연속 길이 (킬 간격 gap 스텝 이내 유지).

    min_r < reward <= max_r 인 스텝을 킬로 간주.
    목숨이 줄어들면 streak 초기화.
    """
    streak = 0
    max_s = 0
    last_kill_step = -1
    prev_lives = None

    for d in data:
        lives = d.get('lives')
        if lives is not None and prev_lives is not None and lives < prev_lives:
            streak = 0
            last_kill_step = -1
        prev_lives = lives if lives is not None else prev_lives

        r = d.get('reward', 0)
        if min_r < r <= max_r:
            g = d['step'] - last_kill_step if last_kill_step >= 0 else gap + 1
            streak = streak + 1 if g <= gap else 1
            max_s = max(max_s, streak)
            last_kill_step = d['step']

    return max_s


def life_losses(data: list) -> int:
    """에피소드 전체 목숨 손실 횟수 (줄어든 총합)."""
    vals = [d['lives'] for d in data if d.get('lives') is not None]
    return sum(
        max(0, vals[i - 1] - vals[i])
        for i in range(1, len(vals))
    )


def survival_steps(data: list) -> int:
    """에피소드 전체 스텝 수."""
    return len(data)
