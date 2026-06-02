"""
ai_agents/space_invaders/coach_config.py
─────────────────────────────────────────
Space Invaders 코칭 설정.
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

역할:
- 플레이어님이 한 행동과 AI 에이전트가 선택한 행동을 비교하고, 에이전트의 선택이 왜 더 좋은 판단이었는지 설명합니다.
- 그 상황에서 어떻게 행동해야 했는지 플레이어님이 다음 판에 바로 써먹을 수 있는 조언을 줍니다.
- 분석 보고서가 아니라, 옆에서 직접 말해주는 코치처럼 자연스럽게 설명하세요.

게임 정보:
- 플레이어님은 화면 아래에서 좌우로 움직이며 위쪽 적들을 쏴 맞춥니다.
- 적이 줄수록 이동 속도가 빨라집니다. 방패는 탄을 막아주며 점차 소모됩니다.
- 모든 적을 처치하면 스테이지가 클리어되고 다음 스테이지로 넘어갑니다(게임오버가 아닙니다).
- 방패는 한 번 손상되면 절대 회복되지 않습니다. "방패를 회복한다"는 표현은 절대 쓰지 마세요.
- 주요 행동: NOOP(대기), FIRE(발사), LEFT(왼쪽), RIGHT(오른쪽), LEFTFIRE(왼쪽+발사), RIGHTFIRE(오른쪽+발사)

코칭 원칙:
- 플레이어님의 행동과 에이전트 행동을 반드시 대비해서 설명하세요.
- "에이전트가 왜 더 좋은 선택이었는가"를 그 상황의 맥락으로 설명하세요.
- 마지막은 반드시 "다음에 이런 상황이라면..." 스타일의 실천 조언으로 마무리하세요.
- 영어 표현을 섞지 마세요.
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
            return "마지막 1마리가 남아 이동 속도가 최고조입니다. 빠르게 움직이는 적을 끝까지 추적해 신속히 처리해야 합니다."
        return f"마지막 {ec}마리가 남아 이동 속도가 매우 빠른 단계입니다. 빠른 적의 움직임을 예측해 신속히 처리하는 것이 핵심입니다."
    if phase == 'fast':
        return f"적이 {ec}마리로 줄어 속도가 빨라지기 시작한 단계입니다. 빠른 적의 궤도를 미리 읽고 공격 타이밍을 잡는 것이 중요합니다."
    if ec >= 35:
        return "적이 빽빽이 밀집한 초반 단계입니다. 과도한 이동보다 적극적으로 적을 많이 제거하는 데 집중하는 것이 유리합니다."
    return "적이 어느 정도 줄어든 중반 단계입니다. 공격과 위치 선정을 균형 있게 가져가는 것이 중요합니다."


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
    shields    = gs.get('shield_integrity', [])
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
        return (f"이 비교 구간에서 플레이어님은 격추되었지만 에이전트는 생존을 유지했습니다. "
                f"생존 여부 자체가 이 순간 두 행동의 결과 차이를 보여줍니다. "
                f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수도 기록했습니다. "
                f"격추 직전 상황과 행동 차이가 어떻게 생존 여부로 이어졌는지 설명하세요.")
    if h_died and a_died:
        if a_score > h_score:
            return (f"이 비교 구간에서 두 경로 모두 격추되었지만, "
                    f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                    f"생존 여부는 같지만 점수 차이가 이 순간 두 행동의 결과 차이를 보여줍니다. "
                    f"점수 차이와 Q값 차이({gap:.4f})를 함께 근거로 활용해 에이전트의 행동이 왜 더 나은 판단이었는지 설명하세요.")
        return (f"이 비교 구간에서 두 경로 모두 격추되었습니다. "
                f"에이전트({a_score:.0f}점)와 플레이어님({h_score:.0f}점)의 점수 차이가 크지 않으므로 "
                f"점수를 결정적 근거로 쓰지 마세요. "
                f"대신 이 순간의 Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하고, "
                f"'이 상황 자체가 어느 쪽으로도 생존이 매우 어려운 국면이었다'는 점을 솔직하게 인정한 뒤 "
                f"그럼에도 에이전트의 행동이 왜 더 나은 판단이었는지 Q값 근거로 설명하세요.")
    if not h_died and a_died:
        return (f"이 비교 구간에서 에이전트 경로가 격추되었고 플레이어님은 생존했습니다. "
                f"단기 결과로는 플레이어님의 행동이 생존 면에서 앞섰습니다. "
                f"이 점을 솔직하게 인정하되, Q값은 에이전트 행동({summary.get('agent_q', 0):.4f})을 더 높게 평가한다는 점과 "
                f"Q값이 수천 번의 학습에서 통계적으로 더 유리한 위치를 만드는 판단을 반영한다는 점을 설명하세요. "
                f"이 특정 구간의 결과가 Q값의 장기 판단과 항상 일치하지 않을 수 있음도 자연스럽게 인정하세요.")
    if a_score > h_score:
        return (f"이후 약 30초 비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                f"이후 득점 프레임 데이터를 활용해 두 경로가 어떻게 달라졌는지 구체적으로 묘사하세요.")
    return (f"이후 약 30초 비교 구간에서는 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
            f"대신 이 순간의 Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")


# ── user 프롬프트 ─────────────────────────────────────────────────────────────

def build_user_prompt(summary: dict[str, Any]) -> str:
    gs         = summary.get('game_state', {}) or {}
    gs_section = _game_state_flags(gs)
    phase_note = _game_phase_note(gs)
    ec         = gs.get('enemy_count', 55)
    a_first    = summary.get('agent_first_reward_step')
    stage_note = (
        "\n※ 이 순간 남은 적이 극소수이고 에이전트 쪽 첫 득점이 매우 빠릅니다. "
        "이는 스테이지 마지막 적을 처치하며 스테이지가 클리어된 상황일 수 있습니다. "
        "득점을 '즉시 획득'처럼 표현하기보다, 스테이지 종료 시점의 위치 선택과 행동의 전략적 차이를 설명하세요."
        if ec <= 3 and isinstance(a_first, (int, float)) and a_first <= 15 else ""
    )
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)

    return f"""다음은 Space Invaders 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 행동의 Q값: {summary.get('human_q', 0):.4f}
- 에이전트 행동의 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 이후 30초 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 첫 득점 시점: {summary.get('human_first_reward_step', '득점 없음')}
- 에이전트 쪽 첫 득점 시점: {summary.get('agent_first_reward_step', '득점 없음')}
- 플레이어님 쪽 득점 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 득점 프레임들: {summary.get('agent_reward_steps') or '없음'}
- 게임 단계 맥락: {phase_note}
{gs_section}
이후 경로 활용 지침: {outcome_guid}{stage_note}

작성 방식:
- 문단 수는 2~3개로 자유롭게 구성하세요. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호(예: "분석", "1." 등)는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 첫 번째 문단: 플레이어님 행동과 에이전트 행동을 명확히 대비하고, "이후 경로 활용 지침"에 따라 이후 전개를 설명하세요. "주목할 상황"이 제공된 경우, 나열하지 말고 배경으로 자연스럽게 녹이세요.
- 중간 문단(선택): 에이전트의 행동이 왜 그 상황에서 더 좋은 판단이었는지 전략적 이유를 충분히 설명하세요. Q값 차이의 의미, 위치·속도·위협 요소 등을 자연스럽게 활용하세요.
- 마지막 문단: "다음에 이런 상황이라면" 또는 그와 비슷한 뉘앙스로 시작해 플레이어님이 바로 실천할 수 있는 구체적인 조언을 주세요.
- 점수와 득점 프레임은 전체 게임이 아닌 이 순간 이후 약 30초 비교 구간의 수치입니다. 점수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- Q값 수치를 언급할 때 "에이전트가 더 높은 기대 보상을 가지고 있었음을 보여줍니다"처럼 당연한 설명은 쓰지 마세요. Q값은 수치만 언급하거나, 그 차이가 전략적으로 의미하는 바를 자연스럽게 녹이는 방식으로만 쓰세요.
- 한국어로만 쓰세요. 영어·일본어 등 외국어 표현은 쓰지 마세요.
- 말투는 "했습니다", "좋았습니다", "유리했습니다"처럼 단정한 존댓말로 쓰세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
"""


# ── 폴백 피드백 ───────────────────────────────────────────────────────────────

def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "득점이 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "득점이 없었습니다"
    return (
        f"이 상황에서는 플레이어님이 {h_ko}을 선택했지만, 에이전트의 {a_ko}가 더 유리했습니다. "
        f"비교 영상을 보면 에이전트 쪽은 이동보다 공격 타이밍을 먼저 살리면서 득점 흐름을 앞당겼고, 플레이어님 쪽은 위험을 피하는 데는 성공했지만 "
        f"득점으로 이어지는 기회를 더 늦게 잡았습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 득점은 {h_txt}에 나왔고, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 회피보다 발사 타이밍을 먼저 확보하는 쪽을 우선 고려해보세요."
    )
