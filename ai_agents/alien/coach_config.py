"""ai_agents/alien/coach_config.py — Alien 코칭 설정"""
from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":          "대기",
    "FIRE":          "발사",
    "UP":            "위로 이동",
    "RIGHT":         "오른쪽 이동",
    "LEFT":          "왼쪽 이동",
    "DOWN":          "아래로 이동",
    "UPRIGHT":       "오른쪽 위로 이동",
    "UPLEFT":        "왼쪽 위로 이동",
    "DOWNRIGHT":     "오른쪽 아래로 이동",
    "DOWNLEFT":      "왼쪽 아래로 이동",
    "UPFIRE":        "위로 이동하며 발사",
    "RIGHTFIRE":     "오른쪽으로 이동하며 발사",
    "LEFTFIRE":      "왼쪽으로 이동하며 발사",
    "DOWNFIRE":      "아래로 이동하며 발사",
    "UPRIGHTFIRE":   "오른쪽 위로 이동하며 발사",
    "UPLEFTFIRE":    "왼쪽 위로 이동하며 발사",
    "DOWNRIGHTFIRE": "오른쪽 아래로 이동하며 발사",
    "DOWNLEFTFIRE":  "왼쪽 아래로 이동하며 발사",
}

reward_label = "득점"

def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)

SYSTEM_PROMPT = """당신은 Alien 플레이어님에게 1:1 코칭을 해주는 게임 코치입니다.

게임 정보:
- 플레이어님은 미로 형태의 우주선 내부를 이동하며 외계인을 피하거나 발사체로 처치합니다.
- 에그(알)를 밟아 제거하면 보너스 점수를 얻고, 특수 아이템을 먹으면 일시적으로 무적 상태가 됩니다.
- 외계인에게 닿으면 목숨을 잃습니다. 이동과 발사를 동시에 처리하는 복합 액션이 핵심입니다.
- 주요 행동: NOOP(대기), FIRE(발사), UP/DOWN/LEFT/RIGHT(이동),
  UPFIRE/DOWNFIRE/LEFTFIRE/RIGHTFIRE(이동+발사) 등 18가지
"""

def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (f"플레이어님은 목숨을 잃었지만 에이전트는 생존했습니다. "
                f"에이전트({a_score:.0f}점) vs 플레이어님({h_score:.0f}점). "
                f"이동 경로와 발사 타이밍 차이가 생존에 어떤 영향을 미쳤는지 설명하세요.")
    if h_done and a_done:
        return (f"두 경로 모두 목숨을 잃었습니다. "
                f"Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")
    if not h_done and a_done:
        return (f"에이전트 경로가 목숨을 잃었고 플레이어님은 생존했습니다. "
                f"이 점을 인정하되 Q값({summary.get('agent_q', 0):.4f})이 "
                f"통계적으로 더 유리한 판단을 반영함을 설명하세요.")
    if a_score > h_score:
        return (f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                f"득점 프레임 데이터를 활용해 두 경로의 차이를 묘사하세요.")
    return (f"플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"구간 결과보다 Q값 차이({gap:.4f})의 전략적 의미에 집중하세요.")

def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)
    return f"""다음은 Alien 코칭 사례입니다.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 Q값: {summary.get('human_q', 0):.4f} / 에이전트 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 비교 구간 플레이어님 점수: {h_score:.1f} / 에이전트 점수: {a_score:.1f}
- 플레이어님 첫 득점: {summary.get('human_first_reward_step', '없음')} / 에이전트 첫 득점: {summary.get('agent_first_reward_step', '없음')}
- 득점 프레임 — 플레이어님: {summary.get('human_reward_steps') or '없음'} / 에이전트: {summary.get('agent_reward_steps') or '없음'}

이후 경로 지침: {outcome_guid}
"""

def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "득점이 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "득점이 없었습니다"
    situation  = f"이 순간 플레이어님은 {h_ko}을 선택했지만, 에이전트는 {a_ko}를 택했습니다."
    comparison = (
        "에이전트는 이동과 발사를 동시에 처리하는 복합 액션으로 외계인을 처치하면서 "
        f"득점 흐름을 앞당겼습니다. "
        f"플레이어님 쪽 득점은 {h_txt}에, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다."
    )
    advice = "다음에 이런 상황이라면, 이동 단독보다 이동+발사 복합 액션을 먼저 고려해보세요."
    return f"[상황]\n{situation}\n\n[비교]\n{comparison}\n\n[조언]\n{advice}"