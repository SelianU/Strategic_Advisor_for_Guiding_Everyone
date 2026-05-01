"""
ai_agents/breakout/coach_config.py
────────────────────────────────────
Breakout 코칭 설정.
"""

from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP": "아무것도 하지 않기",
    "FIRE": "공 발사",
    "RIGHT": "오른쪽 이동",
    "LEFT":  "왼쪽 이동",
}

reward_label = "득점"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Breakout 플레이어님에게 1:1 코칭을 해주는 게임 코치입니다.

게임 정보:
- 플레이어님은 화면 아래 패들을 좌우로 움직여 공을 튕기고, 위쪽 벽돌을 모두 부수면 됩니다.
- 공을 놓치면 목숨을 잃습니다. 패들을 공 궤도에 미리 맞추는 것이 핵심입니다.
- 주요 행동: NOOP(대기), FIRE(공 발사), RIGHT(오른쪽), LEFT(왼쪽)
"""


def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (f"플레이어님은 공을 놓쳐 목숨을 잃었지만 에이전트는 랠리를 유지했습니다. "
                f"에이전트({a_score:.0f}점) vs 플레이어님({h_score:.0f}점). "
                f"패들 위치 차이가 어떻게 공을 받아내는 데 영향을 미쳤는지 설명하세요.")
    if h_done and a_done:
        return (f"두 경로 모두 공을 놓쳐 목숨을 잃었습니다. "
                f"Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")
    if not h_done and a_done:
        return (f"에이전트 경로가 공을 놓쳤고 플레이어님은 랠리를 유지했습니다. "
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

    return f"""다음은 Breakout 코칭 사례입니다.

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
    return (
        f"[상황]\n"
        f"이 순간 플레이어님은 {h_ko}을 선택했지만, 에이전트는 {a_ko}를 택했습니다.\n\n"
        f"[비교]\n"
        f"공의 궤도에 먼저 패들을 맞춰두는 에이전트의 움직임이 벽돌에 더 빠르게 공을 보내는 데 효과적이었습니다. "
        f"플레이어님 쪽 득점은 {h_txt}에, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다.\n\n"
        f"[조언]\n"
        f"다음에 이런 상황이라면, 공이 어디로 튈지를 먼저 예측하고 패들을 미리 이동시키는 습관을 들여보세요."
    )