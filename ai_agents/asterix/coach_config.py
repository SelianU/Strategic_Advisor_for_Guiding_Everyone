"""ai_agents/asterix/coach_config.py — Asterix 코칭 설정"""
from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":      "대기",
    "UP":        "위로 이동",
    "RIGHT":     "오른쪽 이동",
    "LEFT":      "왼쪽 이동",
    "DOWN":      "아래로 이동",
    "UPRIGHT":   "오른쪽 위로 이동",
    "UPLEFT":    "왼쪽 위로 이동",
    "DOWNRIGHT": "오른쪽 아래로 이동",
    "DOWNLEFT":  "왼쪽 아래로 이동",
}

reward_label = "득점"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Asterix 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조와 핵심 규칙:
- 아스테릭스는 화면을 가로지르는 로마 병사들 사이를 이동하며 아이템을 획득합니다.
- 로마 병사들은 왼쪽 또는 오른쪽 방향으로 줄지어 행진합니다. 병사와 접촉하면 즉시 목숨을 잃습니다.
- 아이템(소시지, 방패, 투구, 생선 등)은 로마 병사 행렬 속에 섞여 있습니다. 아스테릭스가 직접 위치를 이동해 아이템에 닿아야 획득합니다.
- 아이템 종류는 라운드가 올라갈수록 바뀝니다. 1라운드에서 소시지였던 아이템이 이후 라운드에서는 방패나 항아리 등으로 교체됩니다.
- 라운드가 진행될수록 로마 병사 밀도가 높아지고 이동 속도도 빨라집니다.
- 발사(FIRE) 액션이 없으며, 오직 8방향 이동으로만 아이템을 획득하고 생존합니다.

득점 원리:
- 아이템에 닿을 때마다 즉시 점수가 올라갑니다.
- 아이템을 연속으로 빠르게 획득하면 더 높은 점수를 기록할 수 있습니다.
- 대기(이동 없음)는 득점 기회를 직접 포기하는 것과 같습니다.

이동 전략의 핵심:
- 직선 이동(위/아래/좌/우)보다 대각선 이동이 두 방향을 동시에 커버해 위험 회피와 아이템 접근을 동시에 처리합니다.
- 로마 병사 행렬 사이의 빈 틈(갭)을 미리 읽고 그 방향으로 대각선 이동하는 것이 핵심 기술입니다.
- 아이템의 위치를 기준으로 어느 방향 대각선이 가장 짧은 경로인지 판단해야 합니다.

코칭 방향:
- 단순히 두 행동을 나열하는 것이 아니라, 그 순간의 로마 병사 흐름과 아이템 위치를 기준으로 "에이전트가 왜 그 방향을 선택했는가"를 설명하세요.
- 대기와 이동의 차이가 실제로 득점 타이밍에 얼마나 영향을 주는지 구체적으로 설명하세요.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, DOWNLEFT 같은 영어 약어는 절대 쓰지 마세요.
- 마지막은 "다음에 이런 상황이라면" 스타일의 실천 조언으로 마무리하세요.
"""


def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (f"이 비교 구간에서 플레이어님은 로마 병사와 접촉해 목숨을 잃었지만 에이전트는 생존했습니다. "
                f"에이전트({a_score:.0f}점) vs 플레이어님({h_score:.0f}점). "
                f"두 경로의 이동 방향 선택이 어떻게 생존 여부를 갈랐는지 설명하세요.")
    if h_done and a_done:
        return (f"이 비교 구간에서 두 경로 모두 목숨을 잃었습니다. "
                f"점수 차이({h_score:.0f}점 vs {a_score:.0f}점)보다 Q값 차이({gap:.4f})가 보여주는 전략적 판단 차이에 집중하세요.")
    if not h_done and a_done:
        return (f"이 비교 구간에서 에이전트 경로가 목숨을 잃었고 플레이어님은 생존했습니다. "
                f"단기 생존은 플레이어님이 앞섰지만, Q값은 에이전트 행동({summary.get('agent_q', 0):.4f})을 더 높게 평가했습니다. "
                f"장기적으로 더 많은 아이템을 획득할 수 있는 위치를 점유하는 것이 Q값의 의미임을 설명하세요.")
    if a_score > h_score:
        return (f"이후 약 30초 비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                f"에이전트가 더 일찍, 더 자주 아이템 위치로 이동할 수 있었던 이유를 설명하세요.")
    return (f"이후 약 30초 비교 구간에서 점수는 플레이어님({h_score:.0f}점)과 에이전트({a_score:.0f}점)가 비슷하거나 플레이어님이 앞섭니다. "
            f"점수를 직접 비교 근거로 쓰지 말고, Q값 차이({gap:.4f})가 의미하는 위치 선점 전략의 차이를 설명하세요.")


def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)

    return f"""다음은 Asterix 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 Q값: {summary.get('human_q', 0):.4f} / 에이전트 Q값: {summary.get('agent_q', 0):.4f} / 가치 차이: {summary.get('gap', 0):.4f}
- 이후 30초 비교 구간 — 플레이어님: {h_score:.1f}점 / 에이전트: {a_score:.1f}점
- 플레이어님 득점 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 득점 프레임들: {summary.get('agent_reward_steps') or '없음'}

이후 경로 활용 지침: {outcome_guid}

작성 규칙:
- 2~3개 문단으로 구성. 각 문단 사이 빈 줄 하나.
- 문단 제목·번호·레이블 사용 금지. 바로 본문 시작.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, LEFT, DOWNLEFT 같은 영어 약어는 절대 쓰지 마세요.
- 첫 문단: 그 순간 로마 병사 흐름과 아이템 위치 문맥을 바탕으로, 에이전트의 이동 방향이 왜 그 상황에서 더 나은 선택이었는지 설명하세요. "이후 경로 활용 지침"에 따라 이후 전개도 포함하세요.
- 중간 문단(선택): 대각선 이동의 전략적 이점, 병사 밀집 구역 우회, 아이템 접근 각도 등 게임 메커니즘 관점의 분석을 추가하세요.
- 마지막 문단: "다음에 이런 상황이라면" 뉘앙스로 시작해 즉시 실천 가능한 조언으로 마무리하세요.
- 점수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- 숫자 나열보다 "병사 사이 빈 틈을 노린", "아이템 쪽으로 먼저 방향을 잡은" 같은 구체적 표현을 사용하세요.
- 한국어로만 쓰세요.
- "했습니다", "좋았습니다" 같은 단정한 존댓말 사용.
- 인삿말, 이모티콘, 과한 감탄사 사용 금지.
"""


def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "득점이 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "득점이 없었습니다"
    return (
        f"이 상황에서 플레이어님은 {h_ko}을 선택했지만, 에이전트의 {a_ko}가 더 유리했습니다. "
        f"에이전트는 로마 병사 행렬 사이의 빈 틈 방향으로 즉시 이동해 아이템 위치에 더 빠르게 접근했습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 득점은 {h_txt}에, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 대기보다는 아이템 쪽으로 가장 짧게 연결되는 대각선 방향을 먼저 선택해보세요."
    )
