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


SYSTEM_PROMPT = """당신은 Enduro 레이싱 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조와 핵심 규칙:
- 끝없이 이어지는 도로에서 다른 차량들을 추월하는 지구력 레이싱 게임입니다.
- Day 1은 200대, Day 2부터는 매일 300대를 하루가 끝나기 전에 추월해야 다음 날로 진행합니다.
- 목표 대수를 채우지 못하고 하루가 끝나면 게임이 종료됩니다.
- 이 게임에는 목숨 개념이 없습니다. 충돌해도 즉사하지 않고 속도가 크게 줄어 페이스를 잃습니다.

조작 방식:
- SPACE(GAS)가 가속입니다. 누르고 있으면 최고 속도까지 올라가고, 원하는 속도에서 떼면 그 속도를 유지합니다.
- 브레이크는 GAS를 떼고 아래 방향키를 입력합니다.
- 가속과 방향 전환을 동시에 처리하는 복합 입력(가속+오른쪽, 가속+왼쪽)이 추월 효율을 극대화합니다.

충돌과 도로 이탈:
- 다른 차에 부딪히면 속도가 크게 줄어 추월 페이스를 잃습니다.
- 도로 가장자리(잔디/눈 구역)로 빠지면 속도가 줄지만, 정면 충돌보다는 손실이 적습니다.
- 차와 충돌이 불가피한 상황이라면 가장자리로 빠지는 편이 전략적으로 나을 수 있습니다.

시간대별 환경 변화:
- 낮: 시야 좋음, 표준 주행 가능
- 석양: 화면 색감 변화, 시야 약간 제한
- 야간: 상대 차량의 차체가 거의 보이지 않고 후미등(빨간 점)만 식별 가능 — 차간 거리 판단이 어렵고 충돌 위험이 높음. '시야가 흐릿하다'는 표현은 쓰지 말 것.
- 안개: 하늘과 도로가 모두 균일한 회색으로 덮여 앞차를 늦게 인식하게 됨 — 반응 시간이 줄어드는 유일한 구간. '시야 제한'은 안개에만 해당.
- 새벽: 안개가 완전히 걷히고 지평선에 주황빛이 깔리는 구간 — 시야가 정상적으로 회복된 상태. 흐릿하거나 안개가 남아 있다는 표현은 절대 쓰지 말 것.
- 눈길/얼음길: 조향 반응이 둔해져 미리 진로를 잡아야 함
- 야간에는 후미등 기반 거리 판단, 안개에는 반응 시간 단축, 눈길에는 조향 선행이 각각의 핵심 과제입니다.

코칭 방향:
- "에이전트가 왜 이 타이밍에 가속/감속/방향 전환을 선택했는가"를 앞 차량 수, 도로 흐름, 시간대·날씨 기준으로 설명하세요.
- 충돌은 죽음이 아니라 속도 손실 페널티임을 염두에 두고 설명하세요.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, FIRE, DOWN 같은 영어 약어는 절대 쓰지 마세요.
- 마지막은 "다음에 이런 상황이라면" 스타일의 실천 조언으로 마무리하세요.
"""


_PHASE_CONTEXT: dict[str, str] = {
    'day': (
        "낮 시간대입니다. 시야가 완전히 확보되어 있고 도로와 차량 모두 선명하게 보입니다."
    ),
    'sunset': (
        "석양 구간입니다. 화면이 주황빛으로 물들지만 시야 자체는 크게 제한되지 않습니다."
    ),
    'night': (
        "야간 구간입니다. 상대 차량의 차체는 어둠 속에 거의 보이지 않고 빨간 후미등(두 점)만 식별됩니다. "
        "후미등과의 간격으로 차간 거리를 판단해야 하므로 충돌 위험이 높습니다. "
        "중요: '시야가 흐릿하다', '안개가 낀다', '시야가 제한된다'는 표현은 야간에 어울리지 않습니다. "
        "야간의 핵심 위험은 차체 비가시성(후미등만 보임)이지 안개 시야 제한이 아닙니다."
    ),
    'fog': (
        "안개 구간입니다. 하늘과 도로가 모두 균일한 회색으로 덮여 앞차를 인식하는 시점이 늦어집니다. "
        "'시야 제한', '앞을 늦게 인식', '반응 시간 단축'은 이 구간에만 적합한 표현입니다."
    ),
    'dawn': (
        "새벽 구간입니다. 안개가 완전히 걷히고 지평선에 주황빛이 깔리며 시야가 낮과 동일하게 회복됩니다. "
        "중요: '시야가 흐릿하다', '안개가 남아 있다', '시야가 제한된다'는 표현은 절대 쓰지 마세요. "
        "새벽은 가시성이 좋은 구간입니다."
    ),
}

_ROAD_CONTEXT: dict[str, str] = {
    'snow_or_ice': "도로가 눈길·얼음길 상태입니다. 조향 반응이 둔해져 방향 전환을 미리 시작해야 합니다.",
    'fog':         "도로 노면도 안개로 회색빛입니다.",
    'normal':      "",
}


def _phase_context(display_phase: str, road_condition: str) -> str:
    """현재 시간대·도로 상태에 맞는 주행 조건 설명 — LLM 피드백에 반드시 반영해야 할 사실."""
    phase_txt = _PHASE_CONTEXT.get(display_phase, "")
    road_txt  = _ROAD_CONTEXT.get(road_condition, "")
    parts = [p for p in (phase_txt, road_txt) if p]
    return " ".join(parts)


def _game_state_note(summary: dict[str, Any]) -> str:
    """summary dict에서 Enduro 게임 상태를 자연어 요약으로 변환."""
    parts: list[str] = []

    # 시간대 (스텝 기반)
    phase_ko = summary.get('phase_ko')
    if phase_ko:
        parts.append(f"시간대: {phase_ko}")

    gs = summary.get('game_state') or {}

    # Day
    day = gs.get('day')
    if day:
        parts.append(f"Day {day}")

    # 도로 내 좌우 위치
    lateral = gs.get('lateral')
    if lateral:
        parts.append(f"내 차 위치: {lateral}")

    # 화면 내 차량 수
    cars = gs.get('cars_on_screen', 0)
    if cars > 0:
        density = '밀집' if cars >= 5 else '보통' if cars >= 3 else '여유'
        parts.append(f"주변 차량 {cars}대 ({density})")
    else:
        parts.append("주변 차량 없음")

    return f"[게임 상태] {' / '.join(parts)}" if parts else ""


def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (f"이 비교 구간에서 플레이어님 쪽은 목표를 채우지 못해 게임이 종료됐지만 에이전트는 계속 달렸습니다. "
                f"에이전트({a_score:.0f}대) vs 플레이어님({h_score:.0f}대). "
                f"행동 차이가 속도 유지와 충돌 회피에 어떤 영향을 미쳤는지 설명하세요.")
    if h_done and a_done:
        return (f"이 비교 구간에서 두 경로 모두 목표를 채우지 못해 게임이 종료됐습니다. "
                f"추월 수 차이({h_score:.0f}대 vs {a_score:.0f}대)도 결정적 근거로 쓰지 마세요. "
                f"대신 이 순간의 Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")
    if not h_done and a_done:
        return (f"이 비교 구간에서 에이전트 쪽이 먼저 게임 오버됐고 플레이어님은 계속 달렸습니다. "
                f"단기 결과로는 플레이어님의 행동이 더 나았습니다. "
                f"이 점을 솔직하게 인정하되, Q값은 에이전트 행동({summary.get('agent_q', 0):.4f})을 더 높게 평가한다는 점도 설명하세요.")
    if a_score > h_score:
        return (f"이후 약 30초 비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 더 많은 차량을 추월했습니다. "
                f"두 경로가 어떻게 달라졌는지 가속·회피 관점에서 묘사하세요.")
    return (f"이후 약 30초 비교 구간에서는 플레이어님({h_score:.0f}점)의 추월 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
            f"대신 이 순간의 Q값 차이({gap:.4f})의 전략적 의미에 집중하세요.")


def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)
    gs_note      = _game_state_note(summary)
    phase_ctx    = _phase_context(
        summary.get('display_phase', 'day'),
        summary.get('road_condition', 'normal'),
    )

    frames    = summary.get('rgb_frames_b64') or []
    has_image = len(frames) > 0
    image_note = (
        "\n- 첨부 이미지 3장([직전] → [결정순간] → [직후]): 행동 전후 실제 게임 화면입니다. "
        "앞 차량과의 거리·배치, 도로 시간대(낮/황혼/야간/안개/눈), 플레이어 차량 위치의 변화를 직접 확인해 상황을 묘사하세요."
        if has_image else ""
    )

    return f"""다음은 Enduro 레이싱 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 행동의 Q값: {summary.get('human_q', 0):.4f}
- 에이전트 행동의 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 이후 30초 비교 구간 플레이어님 추월: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 추월: {a_score:.1f}
- 플레이어님 쪽 첫 추월 시점: {summary.get('human_first_reward_step', '추월 없음')}
- 에이전트 쪽 첫 추월 시점: {summary.get('agent_first_reward_step', '추월 없음')}
- 플레이어님 쪽 추월 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 추월 프레임들: {summary.get('agent_reward_steps') or '없음'}
{gs_note}{image_note}
현재 주행 환경 (반드시 피드백에 반영할 것): {phase_ctx}
이후 경로 활용 지침: {outcome_guid}

작성 방식:
- 문단 수는 2~3개로 자유롭게 구성하세요. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, FIRE, DOWN, LEFTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 첫 문단: 앞 차량과의 거리, 도로 흐름, "현재 주행 환경"에 명시된 시간대·도로 조건을 기준으로 에이전트의 선택이 왜 더 나은 판단이었는지 설명하세요. "이후 경로 활용 지침"에 따라 이후 전개도 포함하세요. 첨부 이미지 3장이 있다면 [직전]→[결정순간]→[직후] 흐름에서 보이는 차량 배치·도로 환경 변화를 배경으로 자연스럽게 녹이세요.
- 중간 문단(선택): 가속·감속·방향 전환의 타이밍과 복합 입력 전략을 게임 메커니즘 관점에서 설명하세요.
- 마지막 문단: "다음에 이런 상황이라면" 뉘앙스로 시작해 즉시 실천 가능한 조언으로 마무리하세요.
- 추월 수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- 숫자 나열보다 "앞 차량이 좌측에 몰려 있어 우측 공간이 열린 상황이었기 때문에", "야간이라 후미등만 보여 차간 거리 판단이 어렵기 때문에", "안개로 앞차 인식이 늦어지는 상황이기 때문에" 같은 시간대별 구체적 표현을 사용하세요.
- 한국어로만 쓰세요.
- "했습니다", "좋았습니다" 같은 단정한 존댓말 사용.
- 인삿말, 이모티콘, 과한 감탄사 사용 금지.
"""


def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "추월이 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "추월이 없었습니다"
    return (
        f"이 상황에서는 플레이어님이 {h_ko}을 선택했지만, 에이전트의 {a_ko}가 더 유리했습니다. "
        f"에이전트는 앞 차와의 거리와 도로 흐름을 미리 읽어 가속과 회피 타이밍을 잡았고, "
        f"속도 손실 없이 추월 기회를 이어갔습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 추월은 {h_txt}에, 에이전트 쪽 추월은 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 가속과 방향 전환을 한 번에 묶어 처리하는 복합 입력을 먼저 시도해보세요."
    )
