"""ai_agents/assault/coach_config.py — Assault 코칭 설정"""
from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":       "대기",
    "FIRE":       "발사",
    "UP":         "위로 이동",
    "RIGHT":      "오른쪽 이동",
    "LEFT":       "왼쪽 이동",
    "RIGHTFIRE":  "오른쪽으로 이동하며 발사",
    "LEFTFIRE":   "왼쪽으로 이동하며 발사",
}

reward_label = "격추"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Assault 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조와 핵심 규칙:
- 화면 아래 좌·중·우 세 포대로 상공에서 접근하는 적기를 격추하는 수비형 슈팅 게임입니다.
- 포대는 좌우로 이동할 수 있으며, 각 포대마다 발사 각도가 다릅니다.
  · 중앙 포대: 거의 수직으로 발사 (화면 중앙 위쪽 대응에 유리)
  · 우측 포대: 대각선 오른쪽 위로 발사 (화면 우측 적에게 유리)
  · 좌측 포대: 대각선 왼쪽 위로 발사 (화면 좌측 적에게 유리)
- 발사체가 없는 상태에서 이동만 하면 득점 기회를 놓치므로 이동과 발사를 동시에 처리하는 것이 효율적입니다.

적기 종류와 득점:
- 소형 UFO(빠르게 이동, 낮은 점수): 움직임 예측이 필요하며 놓치면 포대를 공격합니다.
- 대형 편대 적기: 일정 패턴으로 화면을 가로질러 이동합니다.
- 특수 폭탄 드롭 적기: 하강 폭탄을 투하하며, 지상에 도달하면 포대가 파괴됩니다.
- 포대가 모두 파괴되면 게임이 종료됩니다.

발사 위치 전략:
- 포대를 적기 바로 아래로 이동시킨 뒤 발사하면 명중률이 높아집니다.
- 적기가 이동하는 방향을 보고 예상 위치에 선제적으로 발사하는 '선행 사격'이 핵심 기술입니다.
- 이동과 발사를 동시에 처리하면 더 적은 스텝으로 더 많은 격추를 달성할 수 있습니다.

코칭 방향:
- "에이전트가 왜 이 방향으로 이동하면서 발사했는가"를 적기의 위치와 이동 방향, 발사각도 기준으로 설명하세요.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, FIRE, UP, RIGHTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 마지막은 "다음에 이런 상황이라면" 스타일의 실천 조언으로 마무리하세요.
"""


def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (f"이 비교 구간에서 플레이어님은 목숨을 잃었지만 에이전트는 생존했습니다. "
                f"생존 여부 자체가 이 순간 두 행동의 결과 차이를 보여줍니다. "
                f"에이전트({a_score:.0f}점) vs 플레이어님({h_score:.0f}점). "
                f"포대 위치와 발사 타이밍 차이가 생존에 어떤 영향을 미쳤는지 설명하세요.")
    if h_done and a_done:
        return (f"이 비교 구간에서 두 경로 모두 목숨을 잃었습니다. "
                f"점수 차이({h_score:.0f}점 vs {a_score:.0f}점)도 결정적 근거로 쓰지 마세요. "
                f"대신 이 순간의 Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")
    if not h_done and a_done:
        return (f"이 비교 구간에서 에이전트 경로가 목숨을 잃었고 플레이어님은 생존했습니다. "
                f"단기 결과로는 플레이어님의 행동이 생존 면에서 앞섰습니다. "
                f"이 점을 솔직하게 인정하되, Q값은 에이전트 행동({summary.get('agent_q', 0):.4f})을 더 높게 평가한다는 점과 "
                f"Q값이 통계적으로 더 유리한 판단을 반영함을 설명하세요.")
    if a_score > h_score:
        return (f"이후 약 30초 비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                f"격추 프레임 데이터를 활용해 두 경로의 차이를 묘사하세요.")
    return (f"이후 약 30초 비교 구간에서는 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
            f"대신 이 순간의 Q값 차이({gap:.4f})의 전략적 의미에 집중하세요.")


def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)

    return f"""다음은 Assault 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 행동의 Q값: {summary.get('human_q', 0):.4f}
- 에이전트 행동의 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 이후 30초 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 첫 격추 시점: {summary.get('human_first_reward_step', '격추 없음')}
- 에이전트 쪽 첫 격추 시점: {summary.get('agent_first_reward_step', '격추 없음')}
- 플레이어님 쪽 격추 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 격추 프레임들: {summary.get('agent_reward_steps') or '없음'}

이후 경로 활용 지침: {outcome_guid}

작성 방식:
- 문단 수는 2~3개로 자유롭게 구성하세요. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 첫 번째 문단: 플레이어님 행동과 에이전트 행동을 명확히 대비하고, "이후 경로 활용 지침"에 따라 이후 전개를 설명하세요.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, FIRE, RIGHTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 중간 문단(선택): 적기 위치와 이동 방향을 기준으로 에이전트의 포대 이동 및 발사 각도 선택이 왜 더 나은 전략인지 설명하세요.
- 마지막 문단: "다음에 이런 상황이라면" 뉘앙스로 시작해 즉시 실천 가능한 조언으로 마무리하세요.
- 점수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- 숫자 나열보다 "적기가 화면 오른쪽에서 빠르게 접근하고 있어서", "이동과 발사를 동시에 처리해야 격추 타이밍을 놓치지 않기 때문에" 같은 구체적 표현을 사용하세요.
- 한국어로만 쓰세요.
- "했습니다", "좋았습니다" 같은 단정한 존댓말 사용.
- 인삿말, 이모티콘, 과한 감탄사 사용 금지.
"""


def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "격추가 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "격추가 없었습니다"
    return (
        f"이 상황에서는 플레이어님이 {h_ko}을 선택했지만, 에이전트의 {a_ko}가 더 유리했습니다. "
        f"에이전트는 이동과 발사를 동시에 처리하는 복합 액션으로 적기를 격추하면서 "
        f"득점 흐름을 앞당겼습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 격추는 {h_txt}에, 에이전트 쪽 격추는 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 포대를 이동하는 동시에 발사하는 복합 액션을 우선 고려해보세요."
    )
