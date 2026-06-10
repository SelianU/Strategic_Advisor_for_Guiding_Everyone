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
 
역할:
- 플레이어님이 한 행동과 AI 에이전트가 선택한 행동을 비교하고, 에이전트의 선택이 왜 더 좋은 판단이었는지 설명합니다.
- 그 상황에서 어떻게 행동해야 했는지 플레이어님이 다음 판에 바로 써먹을 수 있는 조언을 줍니다.
- 분석 보고서가 아니라, 옆에서 직접 말해주는 코치처럼 자연스럽게 설명하세요.
 
게임 구조와 핵심 규칙:
- 아틀란티스 도시를 침공하는 적 우주선을 세 개의 포대로 격추해 도시를 지키는 수비형 슈팅 게임입니다.
- 도시에는 여러 구역(건물)이 있으며, 적 우주선이 도시까지 내려오면 해당 구역이 파괴됩니다.
- 모든 도시 구역이 파괴되면 게임이 종료됩니다.
 
적 우주선의 이동 방식 (중요):
- 적은 화면 좌측 또는 우측 최상단에서 등장해 반대쪽 끝까지 수평으로 이동합니다.
- 반대쪽 끝에 도달하면 한 단계 아래 고도에서 반대 방향으로 재등장합니다(지그재그 하강).
  예) 오른쪽 끝 등장 → 왼쪽 끝 도달 → 오른쪽 끝 한 단계 아래에서 재등장
- 이 과정을 약 3회 반복하면 적이 포대를 직접 공격할 수 있는 고도에 도달합니다.
- 적이 공격권에 들어오기 전에 미리 격추하는 것이 핵심 전략입니다.
- 화면 중앙은 적의 이동 경로 중 발사각을 맞추기 가장 쉬운 구간입니다.
 
포대 구조와 발사 각도:
- 세 포대가 화면 아래 고정 위치에 있으며, 각각 고정된 방향으로만 발사합니다.
  · 중앙 포대: 수직으로 위쪽을 향해 발사 → 화면 중앙 부근의 적에게 효과적
  · 우측 포대: 왼쪽 위 대각선으로 발사 → 화면 왼쪽에 위치한 적에게 발사각이 맞음
  · 좌측 포대: 오른쪽 위 대각선으로 발사 → 화면 오른쪽에 위치한 적에게 발사각이 맞음
- 좌측·우측 포대는 각각 반대 방향을 향하기 때문에, 적이 끝에서 재등장할 타이밍에 미리 발사하거나
  적의 이동 궤도를 예측해 발사하는 것이 중요합니다.
- 포대 선택 원칙: 현재 적의 x 위치(화면 좌/중/우)를 기준으로 발사각이 맞는 포대를 선택하세요.
 
적 우주선 종류와 위협 수준:
- 고곤(소형 폭격기): 가장 빠르게 하강하며 폭탄을 투하해 포대를 직접 파괴합니다. 격추 시 가장 높은 점수를 줍니다. 최우선 격추 대상입니다.
- 밴디트(중형 공격기): 고곤보다 느리게 하강하며 도시를 집중 공략합니다.
- 타나토이드(대형 모선): 가장 천천히 하강합니다. 격추 점수는 세 종류 중 가장 낮습니다.
- 라운드가 올라갈수록 하강 속도가 빨라지고 다수의 적이 동시에 접근합니다.
 
포대 파괴 순서 (중요):
- 적이 포대를 공격할 때 가운데 포대가 가장 먼저 파괴됩니다.
- 가운데 포대가 없어지면 이후에 자잘한 구조물이 파괴되고, 마지막으로 양쪽 포대가 차례로 파괴됩니다.
- 따라서 가운데 포대를 지키는 것이 게임 지속의 핵심입니다. 고곤이 가운데를 위협할 때 즉각 대응해야 합니다.
 
코칭 원칙:
- 플레이어님의 행동과 에이전트 행동을 반드시 대비해서 설명하세요.
- "에이전트가 왜 이 포대를 선택했는가"를 적의 x 위치(화면 좌/중/우), 고도(도시까지 남은 거리), 적의 종류 기준으로 설명하세요.
- 마지막은 반드시 "다음에 이런 상황이라면..." 스타일의 실천 조언으로 마무리하세요.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, FIRE, RIGHTFIRE, LEFTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 영어 표현을 섞지 마세요.
"""
 
 
# ── 게임 상태 보조 함수 ───────────────────────────────────────────────────────
 
_ZONE_KO = {'left': '화면 왼쪽', 'center': '화면 중앙', 'right': '화면 오른쪽', 'none': ''}
_ALT_KO  = {0: '상단(여유 있음)', 1: '중간 고도', 2: '하강 중(주의)', 3: '도시 근접(위험)'}
_TYPE_KO = {'gorgon': '고곤(소형 폭격기)', 'bandit': '밴디트(중형 공격기)',
            'thanatoid': '타나토이드(대형 모선)', 'none': ''}
 
 
def _game_phase_note(gs: dict[str, Any]) -> str:
    """게임 상태를 한 줄 단계 요약으로 변환."""
    if not gs:
        return ""
    structs     = gs.get('city_structures_remaining', 7)
    alt         = gs.get('most_threatening_altitude', -1)
    etype       = gs.get('most_threatening_type', 'none')
    deathray    = gs.get('deathray_active', False)
    enemy_count = gs.get('enemy_count', 0)
 
    if deathray:
        return "현재 적 폭탄이 낙하 중입니다. 포대 파괴를 막으려면 고곤을 최우선으로 처리해야 합니다."
    if structs is not None and structs <= 2:
        return "도시 구역이 거의 남지 않은 위기 상황입니다. 한 번의 격추 실패가 게임 종료로 이어질 수 있습니다."
    if structs is not None and structs <= 4:
        return "도시 구역이 절반 이하로 줄어든 후반 단계입니다. 고곤 격추와 적 도달 저지가 최우선입니다."
    if alt == 3:
        type_str = _TYPE_KO.get(etype, etype)
        return f"{type_str}이 도시 바로 위까지 내려왔습니다. 즉각 격추하지 않으면 구역이 파괴됩니다."
    if alt == 2:
        return "위협적인 적이 하강 중입니다. 포대 발사각을 맞춰 격추 타이밍을 잡는 것이 중요합니다."
    if enemy_count >= 3:
        return f"적 {enemy_count}기가 동시에 접근 중입니다. 고곤을 먼저 처리하고 나머지를 순서대로 제거하세요."
    return "초·중반 단계입니다. 적이 공격권에 들어오기 전에 미리 격추하는 습관을 들이는 것이 중요합니다."
 
 
def _game_state_flags(gs: dict[str, Any]) -> str:
    """게임 상태에서 주목할 상황 플래그를 추출."""
    if not gs:
        return ""
    flags: list[str] = []
 
    structs = gs.get('city_structures_remaining', -1)
    if 0 <= structs <= 2:
        flags.append(f"도시 구역 {structs}/7개만 남아 위기 상황")
    elif 0 <= structs <= 4:
        flags.append(f"도시 구역 {structs}/7개로 절반 이하")
 
    if gs.get('deathray_active'):
        flags.append("적 폭탄 낙하 중 — 포대 파괴 위험")
 
    enemy_count = gs.get('enemy_count', 0)
    if enemy_count >= 3:
        flags.append(f"적 {enemy_count}기 동시 접근")
 
    alt   = gs.get('most_threatening_altitude', -1)
    etype = gs.get('most_threatening_type', 'none')
    zone  = gs.get('most_threatening_zone', 'none')
    if alt == 3:
        flags.append(f"{_TYPE_KO.get(etype, etype)}이 {_ZONE_KO.get(zone, zone)}에서 도시 바로 위까지 하강")
    elif alt == 2:
        flags.append(f"{_TYPE_KO.get(etype, etype)}이 {_ZONE_KO.get(zone, zone)}에서 하강 중(주의)")
 
    return f"\n주목할 상황: {' / '.join(flags)}." if flags else ""
 
 
# ── 이후 경로 지침 ────────────────────────────────────────────────────────────
 
def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)
 
    if h_done and not a_done:
        return (f"이 비교 구간에서 플레이어님은 도시를 잃었지만 에이전트는 방어에 성공했습니다. "
                f"방어 여부 자체가 이 순간 두 행동의 결과 차이를 보여줍니다. "
                f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수도 기록했습니다. "
                f"포대 선택과 격추 타이밍 차이가 어떻게 방어 결과로 이어졌는지 설명하세요.")
    if h_done and a_done:
        if a_score > h_score:
            return (f"이 비교 구간에서 두 경로 모두 도시를 잃었지만, "
                    f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                    f"점수 차이와 Q값 차이({gap:.4f})를 함께 근거로 활용해 에이전트의 행동이 왜 더 나은 판단이었는지 설명하세요.")
        return (f"이 비교 구간에서 두 경로 모두 도시를 잃었습니다. "
                f"점수 차이가 크지 않으므로 점수를 결정적 근거로 쓰지 마세요. "
                f"대신 이 순간의 Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이에 집중하고, "
                f"이 상황이 어느 쪽으로도 방어가 어려운 국면이었음을 솔직하게 인정한 뒤 "
                f"그럼에도 에이전트의 행동이 왜 더 나은 판단이었는지 설명하세요.")
    if not h_done and a_done:
        return (f"이 비교 구간에서 에이전트 경로가 도시를 잃었고 플레이어님은 방어했습니다. "
                f"단기 결과로는 플레이어님의 행동이 방어 면에서 앞섰습니다. "
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
    gs           = summary.get('game_state', {}) or {}
    gs_section   = _game_state_flags(gs)
    phase_note   = _game_phase_note(gs)
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)
 
    return f"""다음은 Atlantis 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.
 
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
- 게임 단계 맥락: {phase_note}
{gs_section}
이후 경로 활용 지침: {outcome_guid}
 
작성 방식:
- 문단 수는 2~3개로 자유롭게 구성하세요. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호(예: "분석", "1." 등)는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 첫 번째 문단: 플레이어님 행동과 에이전트 행동을 명확히 대비하고, "이후 경로 활용 지침"에 따라 이후 전개를 설명하세요. "주목할 상황"이 제공된 경우, 나열하지 말고 배경으로 자연스럽게 녹이세요.
- 중간 문단(선택): 에이전트의 행동이 왜 그 상황에서 더 좋은 판단이었는지 전략적 이유를 설명하세요. 적의 x 위치(화면 좌/중/우), 고도, 종류(고곤/밴디트/타나토이드), 발사각 일치 여부를 자연스럽게 활용하세요.
- 마지막 문단: "다음에 이런 상황이라면" 또는 그와 비슷한 뉘앙스로 시작해 플레이어님이 바로 실천할 수 있는 구체적인 조언을 주세요.
- 점수와 격추 프레임은 전체 게임이 아닌 이 순간 이후 약 30초 비교 구간의 수치입니다. 점수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- Q값 수치를 언급할 때 당연한 해석을 반복하지 마세요. Q값 차이가 전략적으로 의미하는 바를 자연스럽게 녹이는 방식으로만 쓰세요.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, FIRE, RIGHTFIRE, LEFTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 적이 대각선으로 이동한다는 표현은 절대 쓰지 마세요. 적은 수평으로 이동하면서 끝에 도달할 때마다 한 단계씩 고도가 낮아집니다.
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
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "격추가 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "격추가 없었습니다"
    return (
        f"이 상황에서는 플레이어님이 {h_ko}을 선택했지만, 에이전트의 {a_ko}가 더 유리했습니다. "
        f"에이전트는 현재 적의 위치에 발사각이 맞는 포대를 정확히 선택해 격추 흐름을 앞당겼습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 격추는 {h_txt}에, 에이전트 쪽 격추는 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 적이 화면 어느 쪽에 위치하는지 먼저 확인하고 발사각이 맞는 포대로 즉시 대응해보세요."
    )
