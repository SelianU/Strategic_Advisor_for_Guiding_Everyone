"""ai_agents/assault/coach_config.py — Assault 코칭 설정"""
from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":       "대기",
    "FIRE":       "위쪽 발사",
    "UP":         "위쪽 발사",          # ALE action 2 = UP = 수직 발사
    "RIGHT":      "오른쪽 이동",
    "LEFT":       "왼쪽 이동",
    "RIGHTFIRE":  "오른쪽으로 이동하며 사선 발사",
    "LEFTFIRE":   "왼쪽으로 이동하며 사선 발사",
}

reward_label = "격추"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Assault 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조:
- 화면 아래에 포대(cannon)가 있고, 상공에서 내려오는 적기를 격추하는 슈팅 게임입니다.
- 포대는 좌우로 이동할 수 있습니다.

발사 방향과 액션:
- 위쪽 발사: 포대가 거의 수직으로 발사합니다. 포대 정면 위쪽의 적에게 유효합니다.
- 오른쪽으로 이동하며 사선 발사: 오른쪽 대각선 위로 발사합니다. 오른쪽에 있는 적에게 유효합니다.
- 왼쪽으로 이동하며 사선 발사: 왼쪽 대각선 위로 발사합니다. 왼쪽에 있는 적에게 유효합니다.
- 이동만(오른쪽/왼쪽 이동): 발사 없이 위치만 바꿉니다. 득점 기회를 놓길 수 있습니다.
- 대기: 발사도 이동도 하지 않습니다.

에너지바 관리 (핵심):
- 발사할 때마다 에너지바가 올라갑니다. 에너지바가 100%에 도달하면 목숨을 잃습니다.
- 에너지바가 60% 이상이면 발사를 자제해야 합니다.
- 에너지바가 80% 이상이면 즉시 발사를 멈춰야 합니다.
- 대기 또는 이동만 하면 에너지바가 내려갑니다.

코칭 원칙:
- 게임 상태(적기 위치·포대 위치·에너지바)를 반드시 피드백에 녹이세요.
- 에이전트가 선택한 발사 방향이 왜 그 적에게 유효한지 방향과 위치로 설명하세요.
- 에너지바 상황이 발사/대기 선택에 미친 영향을 설명하세요.
- 행동 이름은 한국어만 사용하세요. NOOP, FIRE, UP, RIGHTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 마지막은 "다음에 이런 상황이라면" 스타일의 실천 조언으로 마무리하세요.
"""


# ── 게임 상태 → 텍스트 변환 ───────────────────────────────────────────────────

def _game_state_note(gs: dict) -> str:
    if not gs:
        return ""
    lines = []

    # 포대 위치
    px_zone = gs.get('player_zone', '')
    px_pct  = gs.get('player_x_pct', '')
    if px_zone:
        lines.append(f"- 포대 위치: 화면 {px_zone} ({px_pct}%)")

    # 적기 현황
    enemies = gs.get('enemies', [])
    if enemies:
        for e in enemies:
            # rel과 zone이 같으면 하나만 표시 (예: "오른쪽 오른쪽" 방지)
            if e['rel'] == e['zone']:
                pos_desc = e['rel']
            else:
                pos_desc = f"{e['rel']} ({e['zone']} 구역)"
            lines.append(
                f"- 적기 {e['slot']+1}번: 포대 기준 {pos_desc} "
                f"(Y={e['y']}, 거리={e['dist']}px)"
            )
        closest = gs.get('closest_enemy')
        if closest:
            if closest['rel'] == closest['zone']:
                c_pos = closest['rel']
            else:
                c_pos = f"{closest['rel']} ({closest['zone']} 구역)"
            lines.append(
                f"- 가장 가까운 적기: {c_pos}, "
                f"거리 {closest['dist']}px, 행 {closest['row']+1} (행 1이 가장 낮음·위협 큼)"
            )
    else:
        lines.append("- 현재 활성 적기 없음")

    # 모선
    ms_zone = gs.get('mothership_zone', '')
    if ms_zone:
        lines.append(f"- 모선 위치: {ms_zone}")

    # 에너지바
    energy_pct    = gs.get('energy_pct')
    energy_status = gs.get('energy_status', '')
    if energy_pct is not None:
        lines.append(f"- 에너지바: {energy_pct}% — {energy_status}")

    # 적 미사일
    if gs.get('enemy_missile_active'):
        lines.append("- 적 미사일 비행 중 (위협 상황)")

    # 내 미사일
    if gs.get('missile_in_flight'):
        lines.append("- 내 미사일 이미 비행 중 (재발사 시 에너지 추가 소모)")

    lines.append(f"- 잔여 목숨: {gs.get('lives', '?')}")

    if not lines:
        return ""
    return "현재 게임 상태:\n" + "\n".join(lines)


# ── 이후 경로 지침 ────────────────────────────────────────────────────────────

def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (
            f"이 비교 구간에서 플레이어님은 목숨을 잃었지만 에이전트는 생존했습니다. "
            f"에이전트({a_score:.0f}점) vs 플레이어님({h_score:.0f}점). "
            f"포대 위치, 발사 방향 선택, 에너지바 관리 중 어떤 차이가 생존 여부로 이어졌는지 설명하세요."
        )
    if h_done and a_done:
        if a_score > h_score:
            return (
                f"이 비교 구간에서 두 경로 모두 목숨을 잃었지만, "
                f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 더 많은 격추를 기록했습니다. "
                f"Q값 차이({gap:.4f})를 근거로 에이전트의 행동이 왜 더 나은 판단이었는지 설명하세요."
            )
        return (
            f"이 비교 구간에서 두 경로 모두 목숨을 잃었으며 점수 차이도 작습니다. "
            f"점수를 결정적 근거로 쓰지 마세요. "
            f"Q값 차이({gap:.4f})가 의미하는 전략적 판단 차이(발사 방향·에너지바 관리)에 집중하세요."
        )
    if not h_done and a_done:
        return (
            f"이 비교 구간에서 에이전트 경로가 목숨을 잃었고 플레이어님은 생존했습니다. "
            f"단기 결과로는 플레이어님의 행동이 생존 면에서 앞섰습니다. "
            f"이 점을 솔직하게 인정하되, Q값은 에이전트 행동({summary.get('agent_q', 0):.4f})을 더 높게 평가한다는 점을 설명하세요."
        )
    if a_score > h_score:
        return (
            f"이후 약 30초 비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 더 많이 격추했습니다. "
            f"격추 프레임 데이터와 게임 상태를 활용해 두 경로가 어떻게 달라졌는지 설명하세요."
        )
    return (
        f"이후 약 30초 비교 구간에서 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
        f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
        f"Q값 차이({gap:.4f})가 의미하는 발사 방향·에너지바 판단 차이에 집중하세요."
    )


# ── user 프롬프트 ─────────────────────────────────────────────────────────────

def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)
    gs           = summary.get('game_state') or {}
    gs_note      = _game_state_note(gs)

    # ── 에이전트 행동 해석 힌트 ────────────────────────────────────────────────
    agent_action  = summary.get('agent_action_name', '')
    closest       = gs.get('closest_enemy') or {}
    energy_pct    = gs.get('energy_pct')
    a_log_sum     = summary.get('agent_log_summary') or {}
    h_log_sum     = summary.get('human_log_summary') or {}

    a_dominant    = a_log_sum.get('dominant', agent_action)
    a_move_counts = a_log_sum.get('move_counts', {})
    h_dominant    = h_log_sum.get('dominant', summary.get('human_action_name', ''))

    _DIR_KO = {
        'RIGHT': '오른쪽', 'LEFT': '왼쪽',
        'RIGHTFIRE': '오른쪽', 'LEFTFIRE': '왼쪽',
        'FIRE': '제자리', 'UP': '제자리', 'NOOP': '제자리',
    }

    a_total = a_log_sum.get('total_frames', 0)
    h_total = h_log_sum.get('total_frames', 0)

    def _move_summary(move_counts, total):
        if not move_counts or not total:
            return ''
        parts = []
        for act, cnt in sorted(move_counts.items(), key=lambda x: -x[1]):
            pct = round(cnt * 100 / total)
            ko  = _ACTION_KO.get(act, act)
            parts.append(f"{ko} {pct}%")
        return ', '.join(parts[:3])

    a_move_txt = _move_summary(a_move_counts, a_total)
    h_move_txt = _move_summary(
        {k: v for k, v in h_log_sum.get('action_counts', {}).items()
         if k not in ('NOOP', 'FIRE', 'UP')}, h_total)
    direction_hint = ""

    if agent_action in ('RIGHTFIRE', 'LEFTFIRE', 'FIRE', 'UP'):
        fire_dir = _DIR_KO.get(agent_action, '')
        if closest:
            ce_pos = closest.get('rel', '')
            ce_zone = closest.get('zone', '')
            ce_dist = closest.get('dist', '')
            if agent_action in ('FIRE', 'UP'):
                direction_hint = (f"에이전트는 포대 {ce_pos} {ce_zone}의 "
                                  f"적기(거리 {ce_dist}px)를 향해 수직 발사했습니다.")
            else:
                direction_hint = (f"에이전트는 {ce_pos}에 있는 적기(거리 {ce_dist}px)를 "
                                  f"겨냥해 {_ACTION_KO.get(agent_action, agent_action)}을 선택했습니다.")
        if a_move_txt:
            direction_hint += f" 이후 {a_total}프레임 동안 {a_move_txt} 행동을 취했습니다."

    elif agent_action in ('RIGHT', 'LEFT', 'NOOP'):
        if energy_pct is not None and energy_pct >= 55:
            direction_hint = (f"에이전트는 에너지바가 {energy_pct}%로 높아 발사 없이 "
                              f"이동하며 에너지바를 낮추는 판단을 했습니다.")
            if a_move_txt:
                direction_hint += f" 이후 {a_total}프레임 동안 {a_move_txt}."
        elif a_move_txt and a_total:
            if closest:
                ce_pos = closest.get('rel', '')
                ce_dist = closest.get('dist', '')
                direction_hint = (f"에이전트는 이후 {a_total}프레임 동안 {a_move_txt} 행동을 취하며 "
                                  f"포대 {ce_pos} 적기(거리 {ce_dist}px)에 대응할 위치를 잡았습니다.")
            else:
                direction_hint = (f"에이전트는 이후 {a_total}프레임 동안 {a_move_txt} 행동을 이어갔습니다.")
        elif agent_action == 'NOOP' and gs.get('missile_in_flight') and closest:
            direction_hint = (f"미사일이 비행 중이고 포대 {closest.get('rel','')}에 "
                              f"적기(거리 {closest.get('dist','')}px)가 있어, "
                              f"에이전트는 대기하며 명중 여부를 확인했습니다.")
    hint_line = f"\n에이전트 행동 해석 (반드시 피드백에 반영할 것): {direction_hint}" if direction_hint else ""

    return f"""다음은 Assault 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 에이전트 행동의 Q값이 플레이어님 행동보다 {summary.get('gap', 0):.2f} 높음
- 이후 30초 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 격추 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 격추 프레임들: {summary.get('agent_reward_steps') or '없음'}

{gs_note}{hint_line}

이후 경로 활용 지침: {outcome_guid}

작성 방식:
- 문단 수는 2~3개. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목·레이블·번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 행동 이름은 반드시 한국어로만 표현하세요. 영어 약어(NOOP, FIRE, UP, RIGHTFIRE 등)는 절대 쓰지 마세요.
- 첫 번째 문단: 플레이어님과 에이전트의 행동을 대비하세요. "에이전트 행동 해석"이 제공된 경우 이를 근거로, 에이전트가 어느 방향의 적기를 노렸고 왜 그 행동을 선택했는지 구체적으로 설명하세요. "이후 경로 활용 지침"에 따라 이후 전개도 포함하세요.
- 중간 문단(선택): 에너지바 수치가 발사·대기 판단에 미친 영향이나, 적기 행 위치(행 1이 가장 낮아 위협 큼)와 발사 타이밍의 관계를 설명하세요.
- 마지막 문단: "다음에 이런 상황이라면" 뉘앙스로 시작해 즉시 실천 가능한 조언으로 마무리하세요.
- 점수를 언급할 때는 "비교 구간에서"라고 명시하세요.
- Q값 수치는 전체 피드백에서 단 한 번만 언급 가능합니다.
- "했습니다", "좋았습니다" 같은 단정한 존댓말 사용. 인삿말·이모티콘·과한 감탄사 금지.
- 한국어로만 쓰세요.
"""


# ── 폴백 피드백 ───────────────────────────────────────────────────────────────

def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    gs      = summary.get('game_state') or {}
    closest = gs.get('closest_enemy') or {}
    energy  = gs.get('energy_status', '')
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "격추가 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "격추가 없었습니다"

    enemy_desc = ""
    if closest:
        enemy_desc = f"당시 가장 가까운 적기는 포대 기준 {closest.get('rel','')} {closest.get('zone','')}에 있었습니다. "

    energy_desc = f"에너지바는 {energy}였습니다. " if energy else ""

    return (
        f"이 상황에서 플레이어님은 {h_ko}을 선택했지만, 에이전트는 {a_ko}를 선택했습니다. "
        f"{enemy_desc}{energy_desc}"
        f"에이전트는 적기 위치에 맞는 발사 방향을 선택하면서 에너지바를 안전 수준으로 유지했습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 격추는 {h_txt}, 에이전트 쪽 격추는 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 적기가 있는 방향으로 이동하면서 발사하는 복합 행동을 우선 고려해보세요."
    )
