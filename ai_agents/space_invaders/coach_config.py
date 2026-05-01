"""
ai_agents/space_invaders/coach_config.py
─────────────────────────────────────────
Space Invaders 코칭 설정.
llm_feedback.py가 이 파일을 동적으로 임포트해 LLM 피드백을 생성합니다.

필수 항목:
    SYSTEM_PROMPT   : LLM 시스템 프롬프트 (str)
    action_name_ko  : 액션 영문명 → 한국어 변환 함수 (callable)
    build_outcome_guidance : 이후 경로 지침 문자열 생성 함수 (callable)
    build_user_prompt      : 최종 user 프롬프트 문자열 생성 함수 (callable)
    build_fallback_feedback: LLM 없을 때 로컬 폴백 텍스트 생성 함수 (callable)

선택 항목:
    reward_label    : 득점 이벤트를 부르는 이름 (기본 "득점")
"""

from __future__ import annotations
from typing import Any

# ── 한국어 액션 이름 ──────────────────────────────────────────────────────────

_ACTION_KO: dict[str, str] = {
    "NOOP":      "아무것도 하지 않기",
    "FIRE":      "발사",
    "LEFT":      "왼쪽 이동",
    "RIGHT":     "오른쪽 이동",
    "LEFTFIRE":  "왼쪽으로 이동하며 발사",
    "RIGHTFIRE": "오른쪽으로 이동하며 발사",
}

reward_label = "득점"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 Space Invaders 플레이어님에게 1:1 코칭을 해주는 게임 코치입니다.

게임 정보:
- 플레이어님은 화면 아래에서 좌우로 움직이며 위쪽 적들을 쏴 맞춥니다.
- 적이 줄수록 이동 속도가 빨라집니다. 방패는 탄을 막아주며 점차 소모됩니다.
- 모든 적을 처치하면 스테이지가 클리어됩니다(게임오버가 아닙니다).
- 방패는 한 번 손상되면 절대 회복되지 않습니다.
- 주요 행동: NOOP(대기), FIRE(발사), LEFT(왼쪽), RIGHT(오른쪽), LEFTFIRE(왼쪽+발사), RIGHTFIRE(오른쪽+발사)
"""


# ── 게임 상태 보조 함수 ───────────────────────────────────────────────────────

def _game_phase_note(gs: dict[str, Any]) -> str:
    if not gs:
        return ""
    ec     = gs.get('enemy_count', 55)
    phase  = gs.get('enemy_speed_phase', 'normal')
    danger = gs.get('danger_distance', 999)
    if danger < 20:
        return "게임오버 직전 극한 상황입니다. 이 순간의 선택이 생존을 결정합니다."
    if danger < 35:
        return "적이 많이 내려온 후반 위기 단계입니다. 생존이 최우선이며 공격보다 위치 확보가 중요합니다."
    if phase == 'critical':
        if ec == 1:
            return "마지막 1마리가 남아 이동 속도가 최고조입니다."
        return f"마지막 {ec}마리가 남아 이동 속도가 매우 빠른 단계입니다."
    if phase == 'fast':
        return f"적이 {ec}마리로 줄어 속도가 빨라지기 시작한 단계입니다."
    if ec >= 35:
        return "적이 빽빽이 밀집한 초반 단계입니다."
    return "적이 어느 정도 줄어든 중반 단계입니다."


def _game_state_flags(gs: dict[str, Any]) -> str:
    if not gs:
        return ""
    flags: list[str] = []
    phase = gs.get('enemy_speed_phase', 'normal')
    ec    = gs.get('enemy_count', 55)
    if phase == 'critical':
        flags.append(f"적이 {ec}마리밖에 남지 않아 이동 속도가 최고 단계")
    elif phase == 'fast':
        flags.append(f"적이 {ec}마리로 줄어 이동 속도가 빨라진 상태")
    danger = gs.get('danger_distance', 999)
    if danger < 20:
        flags.append("적이 머리 바로 위까지 내려와 게임오버 직전")
    elif danger < 35:
        flags.append("적이 위협적으로 가까이 내려온 상태")
    shields   = gs.get('shield_integrity', [])
    meaningful = [p for p in shields if p >= 10]
    if shields and not meaningful:
        flags.append("방패가 완전히 소실되어 무방비 상태")
    elif meaningful:
        avg = sum(meaningful) / len(meaningful)
        if avg < 40:
            flags.append("방패가 심각하게 손상되어 엄폐 효과가 거의 없는 상태")
        elif avg < 70:
            flags.append("방패가 일부 손상된 상태")
    prox               = gs.get('incoming_proximity', 23)
    bullet_over_shield = gs.get('bullet_over_shield', False)
    has_shield         = bool(meaningful)
    if prox < 4:
        flags.append("탄환이 방패 바로 위에 있어 방패가 곧 손상될 위기" if (bullet_over_shield and has_shield)
                     else "탄환이 머리 바로 위로 즉각 회피하지 않으면 격추 위기")
    elif prox < 8:
        flags.append("탄환이 방패 위쪽에 근접한 상태" if (bullet_over_shield and has_shield)
                     else "탄환이 위험 거리까지 근접한 상태")
    return f"\n주목할 상황: {' / '.join(flags)}." if flags else ""


# ── 이후 경로 지침 ────────────────────────────────────────────────────────────

def build_outcome_guidance(summary: dict[str, Any]) -> str:
    gs      = summary.get('game_state', {}) or {}
    ec      = gs.get('enemy_count', 55)
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    def _is_death(done: bool, score: float) -> bool:
        return done and not (ec <= 5 and score > 0)

    h_died = _is_death(h_done, h_score)
    a_died = _is_death(a_done, a_score)

    if h_died and not a_died:
        return (f"플레이어님은 격추되었지만 에이전트는 생존했습니다. "
                f"에이전트({a_score:.0f}점) vs 플레이어님({h_score:.0f}점). "
                f"격추 직전 상황과 행동 차이가 어떻게 생존 여부로 이어졌는지 설명하세요.")
    if h_died and a_died:
        return (f"두 경로 모두 격추되었습니다. 점수 차이를 결정적 근거로 쓰지 마세요. "
                f"Q값 차이({gap:.4f})가 의미하는 전략적 차이에 집중하세요.")
    if not h_died and a_died:
        return (f"에이전트 경로가 격추되었고 플레이어님은 생존했습니다. "
                f"이 점을 인정하되 Q값({summary.get('agent_q', 0):.4f})이 "
                f"통계적으로 더 유리한 판단을 반영함을 설명하세요.")
    if a_score > h_score:
        return (f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                f"득점 프레임 데이터를 활용해 두 경로의 차이를 묘사하세요.")
    return (f"플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"구간 결과보다 Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")


# ── user 프롬프트 ─────────────────────────────────────────────────────────────

def build_user_prompt(summary: dict[str, Any]) -> str:
    gs          = summary.get('game_state', {}) or {}
    gs_section  = _game_state_flags(gs)
    phase_note  = _game_phase_note(gs)
    ec          = gs.get('enemy_count', 55)
    a_first     = summary.get('agent_first_reward_step')
    stage_note  = (
        "\n※ 남은 적이 극소수이고 에이전트 쪽 첫 득점이 매우 빠릅니다. "
        "스테이지 클리어 상황일 수 있으므로 득점을 '즉시 획득'처럼 표현하지 마세요."
        if ec <= 3 and isinstance(a_first, (int, float)) and a_first <= 15 else ""
    )
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)

    return f"""다음은 Space Invaders 코칭 사례입니다.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 Q값: {summary.get('human_q', 0):.4f} / 에이전트 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 비교 구간 플레이어님 점수: {h_score:.1f} / 에이전트 점수: {a_score:.1f}
- 플레이어님 첫 득점: {summary.get('human_first_reward_step', '없음')} / 에이전트 첫 득점: {summary.get('agent_first_reward_step', '없음')}
- 득점 프레임 — 플레이어님: {summary.get('human_reward_steps') or '없음'} / 에이전트: {summary.get('agent_reward_steps') or '없음'}
- 게임 단계: {phase_note}
{gs_section}
이후 경로 지침: {outcome_guid}{stage_note}
"""


# ── 폴백 피드백 ───────────────────────────────────────────────────────────────

def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko   = action_name_ko(summary["human_action_name"])
    a_ko   = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "득점이 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "득점이 없었습니다"
    return (
        f"[상황]\n"
        f"이 순간 플레이어님은 {h_ko}을 선택했지만, 에이전트는 {a_ko}를 택했습니다.\n\n"
        f"[비교]\n"
        f"에이전트는 이동보다 공격 타이밍을 먼저 살리면서 득점 흐름을 앞당겼고, "
        f"플레이어님은 위험을 피하는 데는 성공했지만 득점 기회를 더 늦게 잡았습니다. "
        f"플레이어님 쪽 득점은 {h_txt}에, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다.\n\n"
        f"[조언]\n"
        f"다음에 이런 상황이라면, 회피보다 발사 타이밍을 먼저 확보하는 쪽을 우선 고려해보세요."
    )