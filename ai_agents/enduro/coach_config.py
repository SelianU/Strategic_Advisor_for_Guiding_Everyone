"""
ai_agents/enduro/coach_config.py
──────────────────────────────────
Enduro 코칭 설정.
"""

from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":      "현 속도 유지",
    "FIRE":      "가속",
    "RIGHT":     "오른쪽 이동",
    "LEFT":      "왼쪽 이동",
    "DOWN":      "브레이크(감속)",
    "DOWNRIGHT": "브레이크 + 오른쪽",
    "DOWNLEFT":  "브레이크 + 왼쪽",
    "RIGHTFIRE": "가속 + 오른쪽",
    "LEFTFIRE":  "가속 + 왼쪽",
}

reward_label = "추월"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Enduro 레이싱 플레이어님에게 1:1 코칭을 해주는 게임 코치입니다.

게임 정보:
- 눈보라·안개·야간 등 시야 제한 환경에서 상대 차량을 추월하며 생존하는 레이싱 게임입니다.
- 매 스테이지마다 지정된 수의 차량을 추월해야 다음 단계로 진행합니다.
- 앞 차와 충돌하면 속도가 급격히 줄어 추월 기회를 잃습니다.
- 도로 가장자리 충돌도 속도를 감소시킵니다.
- 주요 행동: NOOP(현 속도 유지), FIRE(가속), RIGHT(오른쪽), LEFT(왼쪽), DOWN(브레이크),
  DOWNRIGHT(브레이크+오른쪽), DOWNLEFT(브레이크+왼쪽), RIGHTFIRE(가속+오른쪽), LEFTFIRE(가속+왼쪽)
"""


def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (f"플레이어님 쪽은 레이스에서 탈락했지만 에이전트는 계속 달렸습니다. "
                f"에이전트({a_score:.0f}점) vs 플레이어님({h_score:.0f}점). "
                f"행동 차이가 속도 유지와 생존에 어떤 영향을 미쳤는지 설명하세요.")
    if h_done and a_done:
        return (f"두 경로 모두 탈락했습니다. "
                f"Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")
    if not h_done and a_done:
        return (f"에이전트 쪽이 탈락했고 플레이어님은 계속 달렸습니다. "
                f"이 점을 인정하되 Q값({summary.get('agent_q', 0):.4f})이 "
                f"통계적으로 더 유리한 판단을 반영함을 설명하세요.")
    if a_score > h_score:
        return (f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 더 많은 차량을 추월했습니다. "
                f"두 경로가 어떻게 달라졌는지 가속·회피 관점에서 묘사하세요.")
    return (f"플레이어님({h_score:.0f}점)의 추월 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"구간 결과보다 Q값 차이({gap:.4f})의 전략적 의미에 집중하세요.")


def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)

    return f"""다음은 Enduro 레이싱 코칭 사례입니다.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 Q값: {summary.get('human_q', 0):.4f} / 에이전트 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 비교 구간 플레이어님 추월: {h_score:.1f} / 에이전트 추월: {a_score:.1f}
- 플레이어님 첫 추월: {summary.get('human_first_reward_step', '없음')} / 에이전트 첫 추월: {summary.get('agent_first_reward_step', '없음')}
- 추월 프레임 — 플레이어님: {summary.get('human_reward_steps') or '없음'} / 에이전트: {summary.get('agent_reward_steps') or '없음'}

이후 경로 지침: {outcome_guid}
"""


def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "추월이 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "추월이 없었습니다"
    return (
        f"[상황]\n"
        f"이 순간 플레이어님은 {h_ko}을 선택했지만, 에이전트는 {a_ko}를 택했습니다.\n\n"
        f"[비교]\n"
        f"에이전트는 앞 차와의 거리와 도로 흐름을 미리 읽어 가속과 회피 타이밍을 잡았고, "
        f"속도 손실 없이 추월 기회를 이어갔습니다. "
        f"플레이어님 쪽 추월은 {h_txt}에, 에이전트 쪽 추월은 {a_txt}에 이어졌습니다.\n\n"
        f"[조언]\n"
        f"다음에 이런 상황이라면, 가속과 방향 전환을 한 번에 묶어 처리하는 복합 입력을 먼저 시도해보세요."
    )