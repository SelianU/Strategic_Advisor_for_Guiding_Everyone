"""ai_agents/asteroids/coach_config.py — Asteroids 코칭 설정"""
from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":          "대기",
    "FIRE":          "발사",
    "UP":            "추진",
    "RIGHT":         "오른쪽 회전",
    "LEFT":          "왼쪽 회전",
    "DOWN":          "방패",
    "UPRIGHT":       "오른쪽 회전하며 추진",
    "UPLEFT":        "왼쪽 회전하며 추진",
    "UPFIRE":        "추진하며 발사",
    "RIGHTFIRE":     "오른쪽 회전하며 발사",
    "LEFTFIRE":      "왼쪽 회전하며 발사",
    "DOWNFIRE":      "방패 사용하며 발사",
    "UPRIGHTFIRE":   "오른쪽 회전하며 추진+발사",
    "UPLEFTFIRE":    "왼쪽 회전하며 추진+발사",
}

reward_label = "격파"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Asteroids 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조와 핵심 규칙:
- 우주 공간에서 우주선을 조종해 떠돌아다니는 소행성과 UFO를 격파하는 슈팅 게임입니다.
- 소행성이나 UFO에 충돌하거나 UFO의 발사체에 맞으면 목숨을 잃습니다.
- 화면 끝에 도달하면 반대편에서 나타납니다(랩핑 화면 구조).

조작 메커니즘 (관성 물리):
- 이 게임의 핵심은 '관성'입니다. 추진을 가하면 그 방향으로 속도가 붙으며, 추진을 멈춰도 즉시 멈추지 않고 계속 이동합니다.
- 회전으로 방향을 바꾼 뒤 추진하면 새 방향으로 가속되지만, 이전 방향의 관성도 남아 있어 합산된 방향으로 이동합니다.
- 따라서 긴급 회피 시 회전만으로는 충분하지 않으며, 추진 타이밍과 방향을 함께 계획해야 합니다.
- 방패는 순간적으로 발사체를 막을 수 있지만 제한이 있습니다.

소행성 분열 메커니즘:
- 큰 소행성(20점)을 격파하면 2개의 중간 소행성(50점)으로 분열됩니다.
- 중간 소행성을 격파하면 2개의 작은 소행성(100점)으로 분열됩니다.
- 작은 소행성만 완전히 파괴하면 제거됩니다.
- 큰 소행성을 격파할 때 분열된 파편이 바로 우주선을 향해 올 수 있어 격파 후 즉시 회피가 필요합니다.

UFO:
- 주기적으로 UFO가 나타나 우주선을 향해 발사체를 쏩니다. UFO 격파는 보너스 점수이지만 위험 부담도 큽니다.

코칭 방향:
- "에이전트가 왜 이 타이밍에 추진/회전/발사를 선택했는가"를 우주선의 현재 속도 벡터, 주변 소행성의 위치와 이동 방향, 분열 후 파편 예상 경로 기준으로 설명하세요.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, UP, FIRE, RIGHTFIRE 같은 영어 약어는 절대 쓰지 마세요.
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
                f"회피 기동과 발사 타이밍 차이가 생존에 어떤 영향을 미쳤는지 설명하세요.")
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
                f"격파 프레임 데이터를 활용해 두 경로의 차이를 묘사하세요.")
    return (f"이후 약 30초 비교 구간에서는 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
            f"대신 이 순간의 Q값 차이({gap:.4f})의 전략적 의미에 집중하세요.")


def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)

    return f"""다음은 Asteroids 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 행동의 Q값: {summary.get('human_q', 0):.4f}
- 에이전트 행동의 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 이후 30초 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 첫 격파 시점: {summary.get('human_first_reward_step', '격파 없음')}
- 에이전트 쪽 첫 격파 시점: {summary.get('agent_first_reward_step', '격파 없음')}
- 플레이어님 쪽 격파 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 격파 프레임들: {summary.get('agent_reward_steps') or '없음'}

이후 경로 활용 지침: {outcome_guid}

작성 방식:
- 문단 수는 2~3개로 자유롭게 구성하세요. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 첫 번째 문단: 플레이어님 행동과 에이전트 행동을 명확히 대비하고, "이후 경로 활용 지침"에 따라 이후 전개를 설명하세요.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, UP, FIRE 같은 영어 약어는 절대 쓰지 마세요.
- 중간 문단(선택): 소행성의 위치·크기·이동 방향과 우주선의 관성 물리를 기준으로 에이전트의 선택이 왜 더 안전하거나 효율적인지 설명하세요.
- 마지막 문단: "다음에 이런 상황이라면" 뉘앙스로 시작해 즉시 실천 가능한 조언으로 마무리하세요.
- 점수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- 숫자 나열보다 "큰 소행성이 바로 앞에서 이동 중이었기 때문에", "격파 후 파편이 우주선 쪽으로 튀는 상황을 예상해서" 같은 구체적 표현을 사용하세요.
- 한국어로만 쓰세요.
- "했습니다", "좋았습니다" 같은 단정한 존댓말 사용.
- 인삿말, 이모티콘, 과한 감탄사 사용 금지.
"""


def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "격파가 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "격파가 없었습니다"
    return (
        f"이 상황에서는 플레이어님이 {h_ko}을 선택했지만, 에이전트의 {a_ko}가 더 유리했습니다. "
        f"에이전트는 회전과 추진, 발사를 정밀하게 조합해 소행성을 격파하면서 "
        f"득점 흐름을 앞당겼습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 격파는 {h_txt}에, 에이전트 쪽 격파는 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 소행성을 향해 먼저 방향을 맞추고 추진과 발사를 동시에 활용해보세요."
    )
