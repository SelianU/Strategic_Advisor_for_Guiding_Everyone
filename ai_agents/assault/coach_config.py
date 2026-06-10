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
 
역할:
- 플레이어님이 한 행동과 AI 에이전트가 선택한 행동을 비교하고, 에이전트의 선택이 왜 더 좋은 판단이었는지 설명합니다.
- 그 상황에서 어떻게 행동해야 했는지 플레이어님이 다음 판에 바로 써먹을 수 있는 조언을 줍니다.
- 분석 보고서가 아니라, 옆에서 직접 말해주는 코치처럼 자연스럽게 설명하세요.
 
게임 구조와 핵심 규칙:
- 우주선을 조종해 화면에 등장하는 적기를 격추하는 슈팅 게임입니다.
- 적기는 우주선(모선)에서 생성되며, 약 3번 하강 패스를 반복한 이후부터 플레이어를 직접 공격합니다. 그 전에 미리 격추하는 것이 안전합니다.
- 적기는 변칙적인 이동 패턴을 가지며, 내 발사체 속도가 빠르지 않으므로 적의 이동 경로를 예측해 선행 사격하는 것이 핵심입니다.
 
에너지바 관리 (중요):
- 발사를 연타하면 에너지바가 차오릅니다. 발사 버튼을 눌러도 실제로 발사되지 않는 경우도 있습니다.
- 에너지바가 가득 차면 목숨을 잃습니다. 발사를 무작정 연타하면 자멸로 이어집니다.
- 필요할 때만 발사하고, 발사 간격을 조절해 에너지바를 안전 수준으로 유지하는 것이 핵심입니다.
 
코칭 원칙:
- 플레이어님의 행동과 에이전트 행동을 반드시 대비해서 설명하세요.
- "에이전트가 왜 이 행동을 선택했는가"를 적기의 위치, 이동 패턴 예측, 발사 타이밍(에너지바 소모 억제) 기준으로 설명하세요.
- 마지막은 반드시 "다음에 이런 상황이라면..." 스타일의 실천 조언으로 마무리하세요.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, FIRE, UP, RIGHTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 영어 표현을 섞지 마세요.
"""


# ── 이후 경로 지침 ────────────────────────────────────────────────────────────

def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (f"이 비교 구간에서 플레이어님은 격추되었지만 에이전트는 생존했습니다. "
                f"생존 여부 자체가 이 순간 두 행동의 결과 차이를 보여줍니다. "
                f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수도 기록했습니다. "
                f"포대 위치와 발사 타이밍 차이가 어떻게 생존 여부로 이어졌는지 설명하세요.")
    if h_done and a_done:
        if a_score > h_score:
            return (f"이 비교 구간에서 두 경로 모두 격추되었지만, "
                    f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                    f"점수 차이와 Q값 차이({gap:.4f})를 함께 근거로 활용해 에이전트의 행동이 왜 더 나은 판단이었는지 설명하세요.")
        return (f"이 비교 구간에서 두 경로 모두 격추되었습니다. "
                f"점수 차이가 크지 않으므로 점수를 결정적 근거로 쓰지 마세요. "
                f"대신 이 순간의 Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")
    if not h_done and a_done:
        return (f"이 비교 구간에서 에이전트 경로가 격추되었고 플레이어님은 생존했습니다. "
                f"단기 결과로는 플레이어님의 행동이 생존 면에서 앞섰습니다. "
                f"이 점을 솔직하게 인정하되, Q값은 에이전트 행동({summary.get('agent_q', 0):.4f})을 더 높게 평가한다는 점과 "
                f"Q값이 수천 번의 학습에서 통계적으로 더 유리한 판단을 반영한다는 점을 설명하세요. "
                f"이 특정 구간의 결과가 Q값의 장기 판단과 항상 일치하지 않을 수 있음도 자연스럽게 인정하세요.")
    if a_score > h_score:
        return (f"이후 약 30초 비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                f"격추 프레임 데이터를 활용해 두 경로가 어떻게 달라졌는지 구체적으로 묘사하세요.")
    return (f"이후 약 30초 비교 구간에서는 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
            f"대신 이 순간의 Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")


# ── user 프롬프트 ─────────────────────────────────────────────────────────────

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
- 문단 제목, 레이블, 번호(예: "분석", "1." 등)는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 첫 번째 문단: 플레이어님 행동과 에이전트 행동을 명확히 대비하고, "이후 경로 활용 지침"에 따라 이후 전개를 설명하세요.
- 중간 문단(선택): 에이전트의 행동이 왜 그 상황에서 더 좋은 판단이었는지 설명하세요. 적기의 이동 패턴 예측, 선행 사격 여부, 발사 간격을 조절해 에너지바 자멸을 피하는 관리 측면을 자연스럽게 활용하세요.
- 마지막 문단: "다음에 이런 상황이라면" 또는 그와 비슷한 뉘앙스로 시작해 플레이어님이 바로 실천할 수 있는 구체적인 조언을 주세요.
- 점수와 격추 프레임은 이 순간 이후 약 30초 비교 구간의 수치입니다. 점수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- Q값 차이가 전략적으로 의미하는 바를 자연스럽게 녹이는 방식으로만 쓰세요. 당연한 해석은 반복하지 마세요.
- 행동 이름은 반드시 한국어로만 표현하세요. 영어 약어는 절대 쓰지 마세요.
- 한국어로만 쓰세요. 말투는 "했습니다", "유리했습니다"처럼 단정한 존댓말로 쓰세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
"""


# ── 폴백 피드백 ───────────────────────────────────────────────────────────────

def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "격추가 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "격추가 없었습니다"
    return (
        f"이 상황에서는 플레이어님이 {h_ko}을 선택했지만, 에이전트의 {a_ko}가 더 유리했습니다. "
        f"에이전트는 적기의 이동 경로를 예측해 최소한의 움직임으로 격추 타이밍을 잡으며 에너지를 효율적으로 관리했습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 격추는 {h_txt}에, 에이전트 쪽 격추는 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 불필요한 이동을 줄이고 적의 이동 방향을 먼저 읽은 뒤 예상 위치에 선행 사격하는 것을 우선 고려해보세요."
    )
