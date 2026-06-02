"""ai_agents/atlantis/coach_config.py — Atlantis 코칭 설정"""
from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":       "대기",
    "FIRE":       "중앙 발사",
    "RIGHTFIRE":  "오른쪽 포대 발사",
    "LEFTFIRE":   "왼쪽 포대 발사",
}

reward_label = "격추"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Atlantis 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조와 핵심 규칙:
- 아틀란티스 도시를 침공하는 적 우주선을 세 개의 포대로 격추해 도시를 지키는 수비형 슈팅 게임입니다.
- 도시에는 여러 구역(건물)이 있으며, 적 우주선이 도시까지 내려오면 해당 구역이 파괴됩니다.
- 모든 도시 구역이 파괴되면 게임이 종료됩니다.

적 우주선의 이동 방식 (중요):
- 적 우주선은 대각선으로 내려오지 않습니다.
- 적은 화면 위쪽에서 좌우로 수평 이동하면서 동시에 아래쪽으로 천천히 내려옵니다(수평+수직 동시 이동).
- 즉, 적은 화면을 가로지르며 이동하는 동안 점점 고도가 낮아지고, 도시 높이까지 내려오면 파괴가 발생합니다.
- 적의 x 위치(화면 어느 쪽에 있는가)가 포대 선택의 핵심 기준입니다.

포대 구조와 발사 각도:
- 세 포대가 화면 아래 고정 위치에 있으며, 각각 고정된 방향으로만 발사합니다.
  · 중앙 포대: 수직으로 위쪽을 향해 발사 → 화면 중앙 부근에 있는 적에게 효과적
  · 우측 포대: 오른쪽 위 대각선으로 발사 → 화면 오른쪽에 위치한 적에게 발사각이 맞음
  · 좌측 포대: 왼쪽 위 대각선으로 발사 → 화면 왼쪽에 위치한 적에게 발사각이 맞음
- 발사체는 직선으로 날아가며, 적의 x 위치가 해당 포대의 발사 궤도에 있어야 명중합니다.
- 포대 선택 원칙: 지금 적이 화면 어느 쪽(왼쪽/중앙/오른쪽)에 있는지를 보고, 그 위치에 발사각이 맞는 포대를 선택해야 합니다.

적 우주선 종류와 위협 수준:
- 대형 모선(타나토이드): 천천히 하강하지만 격추 시 고득점.
- 중형 공격기(밴디트): 더 빠르게 하강하며 도시를 향해 집중 접근.
- 소형 폭격기(고곤): 가장 빠르며 폭탄을 투하해 포대를 직접 파괴합니다.
- 라운드가 올라갈수록 하강 속도가 빨라지고 다수의 적이 동시에 접근합니다.

포대 선택 전략:
- 적이 화면 왼쪽에 있으면 좌측 포대, 오른쪽에 있으면 우측 포대, 중앙이면 중앙 포대가 발사각이 맞습니다.
- 가장 빠르게 도시에 도달하려는 적(고곤 > 밴디트 > 타나토이드 순)에 먼저 대응해야 합니다.
- 대기는 발사 기회를 낭비하는 것과 같으므로 가능한 한 발사를 유지해야 합니다.

코칭 방향:
- "에이전트가 왜 이 포대를 선택했는가"를 적의 x 위치(화면 좌/중/우), 고도(도시까지 남은 거리), 적의 종류 기준으로 설명하세요.
- 절대로 적이 대각선으로 이동한다고 설명하지 마세요. 적은 수평으로 이동하면서 고도가 낮아집니다.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, FIRE, RIGHTFIRE, LEFTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 마지막은 "다음에 이런 상황이라면" 스타일의 실천 조언으로 마무리하세요.
"""


_ZONE_KO = {'left': '화면 왼쪽', 'center': '화면 중앙', 'right': '화면 오른쪽', 'none': ''}
_ALT_KO  = {0: '상단(여유 있음)', 1: '중간 고도', 2: '하강 중(주의)', 3: '도시 근접(위험)'}
_TYPE_KO = {'gorgon': '고곤(소형 폭격기)', 'bandit': '밴디트(중형 공격기)',
            'thanatoid': '타나토이드(대형 모선)', 'none': ''}


def _game_state_note(gs: dict[str, Any]) -> str:
    """RAM 기반 게임 상태를 자연어 한 줄 요약으로 변환."""
    if not gs:
        return ""

    parts: list[str] = []

    # 가장 위협적인 적
    zone  = gs.get('most_threatening_zone', 'none')
    alt   = gs.get('most_threatening_altitude', -1)
    etype = gs.get('most_threatening_type', 'none')
    if zone != 'none' and alt >= 0:
        parts.append(
            f"{_ZONE_KO.get(zone, zone)}에 {_TYPE_KO.get(etype, etype)} 위치 "
            f"({_ALT_KO.get(alt, f'고도 레인 {alt}')})"
        )

    # 도시 구역 잔존
    structs = gs.get('city_structures_remaining', -1)
    if 0 <= structs < 7:
        parts.append(f"도시 구역 {structs}/7개 남음")

    # 폭탄(데스레이) 활성
    if gs.get('deathray_active'):
        parts.append("적 폭탄 낙하 중")

    # 동시 접근 적 수
    enemy_count = gs.get('enemy_count', 0)
    if enemy_count >= 2:
        parts.append(f"적 {enemy_count}기 동시 접근")

    return f"[게임 상태] {' / '.join(parts)}" if parts else ""


def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (f"이 비교 구간에서 플레이어님은 도시를 잃었지만 에이전트는 방어에 성공했습니다. "
                f"방어 여부 자체가 이 순간 두 행동의 결과 차이를 보여줍니다. "
                f"에이전트({a_score:.0f}점) vs 플레이어님({h_score:.0f}점). "
                f"포대 선택과 격추 타이밍 차이가 방어 결과에 어떤 영향을 미쳤는지 설명하세요.")
    if h_done and a_done:
        return (f"이 비교 구간에서 두 경로 모두 도시를 잃었습니다. "
                f"점수 차이({h_score:.0f}점 vs {a_score:.0f}점)도 결정적 근거로 쓰지 마세요. "
                f"대신 이 순간의 Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하세요.")
    if not h_done and a_done:
        return (f"이 비교 구간에서 에이전트 경로가 도시를 잃었고 플레이어님은 방어했습니다. "
                f"단기 결과로는 플레이어님의 행동이 방어 면에서 앞섰습니다. "
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
    gs_note      = _game_state_note(summary.get('game_state') or {})

    return f"""다음은 Atlantis 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 Q값: {summary.get('human_q', 0):.4f} / 에이전트 Q값: {summary.get('agent_q', 0):.4f} / 가치 차이: {summary.get('gap', 0):.4f}
- 이후 30초 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 첫 격추 시점: {summary.get('human_first_reward_step', '격추 없음')}
- 에이전트 쪽 첫 격추 시점: {summary.get('agent_first_reward_step', '격추 없음')}
- 플레이어님 쪽 격추 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 격추 프레임들: {summary.get('agent_reward_steps') or '없음'}
{gs_note}
이후 경로 활용 지침: {outcome_guid}

작성 방식:
- 문단 수는 2~3개로 자유롭게 구성하세요. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, FIRE, RIGHTFIRE, LEFTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 첫 번째 문단: 플레이어님 행동과 에이전트 행동을 대비하되, 핵심은 "에이전트가 왜 그 포대를 선택했는가"입니다. 적 우주선이 어느 방향에서 하강하고 있었는지, 어느 포대의 발사각이 그 경로에 맞는지를 기준으로 설명하세요. "이후 경로 활용 지침"에 따라 이후 전개도 포함하세요.
- 중간 문단(선택): 적 우주선의 종류(타나토이드/밴디트/고곤)와 위협 수준, 도시 구역까지 남은 거리, 발사각 일치 여부를 기준으로 에이전트의 선택이 왜 더 유리했는지 설명하세요.
- 마지막 문단: "다음에 이런 상황이라면" 뉘앙스로 시작해 즉시 실천 가능한 조언으로 마무리하세요.
- 점수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- 숫자 나열보다 "고곤이 화면 왼쪽에 위치해 있어 좌측 포대의 발사각만 맞출 수 있었기 때문에", "도시까지 이미 많이 내려온 밴디트가 화면 오른쪽을 이동 중이어서 우측 포대로 즉시 대응해야 했기 때문에" 같은 구체적 표현을 사용하세요.
- 적이 대각선으로 이동한다는 표현은 절대 쓰지 마세요. 적은 수평 이동 중에 고도가 낮아지는 방식으로 접근합니다.
- 한국어로만 쓰세요.
- "했습니다", "좋았습니다", "유리했습니다"처럼 단정한 존댓말을 사용하세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
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
        f"에이전트는 UFO의 위치에 맞는 포대를 정확히 선택해 "
        f"격추 흐름을 앞당겼습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 격추는 {h_txt}에, 에이전트 쪽 격추는 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, UFO가 어느 방향에서 하강하는지 먼저 파악하고 해당 포대로 즉시 대응해보세요."
    )
