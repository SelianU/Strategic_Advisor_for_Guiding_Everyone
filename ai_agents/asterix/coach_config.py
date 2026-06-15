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

게임 구조:
- 화면은 8개의 가로 레인으로 나뉩니다. 플레이어는 레인을 오르내리며 좌우로 이동합니다.
- 각 레인에는 적(빨간 적군)이나 아이템(항아리)이 흘러옵니다.
- 아이템과 같은 레인, 같은 X 위치를 지나면 자동으로 수집되어 득점합니다.
- 적과 같은 레인에서 충돌하면 목숨을 잃습니다.

핵심 전략:
- 아이템은 레인을 맞추고 지나가기만 해도 수집됩니다. 서두르지 않아도 됩니다.
- 적은 같은 레인 위에 있으면 무조건 충돌합니다. 다른 레인으로 피해야 합니다.
- 한 레인에 아이템이 연속으로 흐를 때는 그 레인에 머무르는 것이 효율적입니다.
- 같은 레인에 적이 있으면 즉시 위아래로 레인을 바꿔야 합니다.

코칭 원칙:
- "에이전트가 왜 그 방향으로 이동했는가"를 플레이어와 객체들의 레인·X 위치로 설명하세요.
- 같은 레인 위협 여부, 가장 가까운 아이템/적의 상대적 위치를 반드시 활용하세요.
- 행동 이름은 한국어만 사용하세요. NOOP, UP, DOWNRIGHT 같은 영어 약어는 절대 쓰지 마세요.
- 마지막은 "다음에 이런 상황이라면" 스타일의 실천 조언으로 마무리하세요.
"""


# ── 게임 상태 → 텍스트 변환 ───────────────────────────────────────────────────

def _game_state_note(gs: dict) -> str:
    if not gs:
        return ""
    lines = []

    p_lane = gs.get('player_lane', '?')
    p_zone = gs.get('player_zone', '')
    lines.append(f"- 플레이어: 레인 {p_lane+1}, {p_zone} (X={gs.get('player_x', '?')})")

    if gs.get('same_lane_threat'):
        lines.append("- ⚠ 현재 레인에 적이 있습니다 (충돌 위험)")

    ce = gs.get('closest_enemy')
    if ce:
        lane_diff = ce['lane_rel']
        dir_str   = f"{abs(lane_diff)}레인 {'위' if lane_diff < 0 else '아래'}" if lane_diff != 0 else "같은 레인"
        lines.append(
            f"- 가장 가까운 적: 레인 {ce['lane']+1}({dir_str}), "
            f"{ce['h_dir']} {ce['zone']} (X={ce['x']}, 거리={ce['dist']}px)"
        )

    cr = gs.get('closest_reward')
    if cr:
        lane_diff = cr['lane_rel']
        dir_str   = f"{abs(lane_diff)}레인 {'위' if lane_diff < 0 else '아래'}" if lane_diff != 0 else "같은 레인"
        lines.append(
            f"- 가장 가까운 아이템: 레인 {cr['lane']+1}({dir_str}), "
            f"{cr['h_dir']} {cr['zone']} (X={cr['x']}, 거리={cr['dist']}px)"
        )

    lines.append(
        f"- 활성 적 {gs.get('enemy_count', 0)}개 / 수집 가능 아이템 {gs.get('reward_count', 0)}개"
    )
    lines.append(f"- 잔여 목숨: {gs.get('lives', '?')}")

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
            f"레인 선택 또는 이동 방향의 차이가 충돌 여부에 어떤 영향을 미쳤는지 설명하세요."
        )
    if h_done and a_done:
        return (
            f"이 비교 구간에서 두 경로 모두 목숨을 잃었습니다. "
            f"점수({h_score:.0f}점 vs {a_score:.0f}점)를 결정적 근거로 쓰지 마세요. "
            f"Q값 차이({gap:.4f})가 의미하는 레인 판단과 회피 전략 차이에 집중하세요."
        )
    if not h_done and a_done:
        return (
            f"이 비교 구간에서 에이전트 경로가 목숨을 잃었고 플레이어님은 생존했습니다. "
            f"단기 결과로는 플레이어님이 앞섰습니다. "
            f"이 점을 인정하되, Q값은 에이전트 행동({summary.get('agent_q', 0):.4f})을 더 높게 평가함을 설명하세요."
        )
    if a_score > h_score:
        return (
            f"이후 약 30초 비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 더 많이 득점했습니다. "
            f"아이템 수집 흐름과 레인 선택 차이를 근거로 설명하세요."
        )
    return (
        f"이후 약 30초 비교 구간에서 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
        f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
        f"Q값 차이({gap:.4f})가 의미하는 레인 전략 차이에 집중하세요."
    )


# ── user 프롬프트 ─────────────────────────────────────────────────────────────

def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)
    gs           = summary.get('game_state') or {}
    gs_note      = _game_state_note(gs)

    # 에이전트 행동 해석 힌트
    agent_action  = summary.get('agent_action_name', '')
    p_lane        = gs.get('player_lane', -1)
    same_threat   = gs.get('same_lane_threat', False)
    ce            = gs.get('closest_enemy') or {}
    cr            = gs.get('closest_reward') or {}

    hint_parts = []

    # 1) 같은 레인 위협 → 위아래 이동이면 회피 목적
    if same_threat and agent_action in ('UP', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT'):
        hint_parts.append(
            f"현재 레인 {p_lane+1}에 적이 있어 충돌 위험 상황입니다. "
            f"에이전트는 레인을 바꿔 적을 피하는 판단을 했습니다."
        )

    # 2) 가장 가까운 아이템 방향과 이동 방향 일치 여부
    if cr:
        lane_rel  = cr.get('lane_rel', 0)
        h_dir     = cr.get('h_dir', '')
        cr_lane   = cr.get('lane', -1)
        cr_dist   = cr.get('dist', 0)
        moving_toward_reward = False
        if lane_rel < 0 and 'UP' in agent_action:
            moving_toward_reward = True
        elif lane_rel > 0 and 'DOWN' in agent_action:
            moving_toward_reward = True
        elif lane_rel == 0 and h_dir == '오른쪽' and 'RIGHT' in agent_action:
            moving_toward_reward = True
        elif lane_rel == 0 and h_dir == '왼쪽' and 'LEFT' in agent_action:
            moving_toward_reward = True

        if moving_toward_reward and not same_threat:
            hint_parts.append(
                f"레인 {cr_lane+1}에 아이템이 있고(거리 {cr_dist}px), "
                f"에이전트는 그쪽으로 이동해 수집 경로를 선점했습니다."
            )

    # 3) 힌트 없으면 기본
    if not hint_parts and ce:
        ce_lane = ce.get('lane', -1)
        ce_dist = ce.get('dist', 0)
        hint_parts.append(
            f"레인 {ce_lane+1}에 적(거리 {ce_dist}px)이 있는 상황에서 "
            f"에이전트는 {action_name_ko(agent_action)}을 선택했습니다."
        )

    direction_hint = " ".join(hint_parts)
    hint_line = f"\n에이전트 행동 해석 (반드시 피드백에 반영할 것): {direction_hint}" if direction_hint else ""

    frames    = summary.get('rgb_frames_b64') or []
    has_image = len(frames) > 0
    image_note = (
        "\n- 첨부 이미지 3장([직전] → [결정순간] → [직후]): 행동 전후 실제 게임 화면입니다. "
        "플레이어 위치, 각 레인의 아이템·적 분포, 이동 방향의 변화를 직접 확인해 상황을 묘사하세요."
        if has_image else ""
    )

    return f"""다음은 Asterix 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 에이전트 행동의 Q값이 플레이어님 행동보다 {summary.get('gap', 0):.2f} 높음
- 이후 30초 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 득점 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 득점 프레임들: {summary.get('agent_reward_steps') or '없음'}

{gs_note}{hint_line}{image_note}

이후 경로 활용 지침: {outcome_guid}

작성 방식:
- 문단 수는 2~3개. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목·레이블·번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, UP, DOWNRIGHT 같은 영어 약어는 절대 쓰지 마세요.
- 첫 번째 문단: 플레이어님과 에이전트의 행동을 대비하세요. "에이전트 행동 해석"이 제공된 경우 이를 근거로, 에이전트가 왜 그 레인·방향을 선택했는지 구체적으로 설명하세요. "이후 경로 활용 지침"에 따라 이후 전개도 포함하세요. 첨부 이미지 3장이 있다면 [직전]→[결정순간]→[직후] 흐름에서 보이는 레인별 아이템·적 분포 변화를 배경으로 자연스럽게 녹이세요.
- 중간 문단(선택): 같은 레인 위협 회피, 가장 가까운 아이템으로의 효율적 경로, 레인 선점 전략을 구체적으로 설명하세요.
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
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "득점이 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "득점이 없었습니다"

    threat  = "현재 레인에 적이 있어 즉시 회피가 필요한 상황이었습니다. " if gs.get('same_lane_threat') else ""
    cr      = gs.get('closest_reward')
    reward_desc = ""
    if cr:
        reward_desc = (
            f"가장 가까운 아이템은 레인 {cr['lane']+1}, {cr['h_dir']} {cr['zone']}에 있었습니다. "
        )

    return (
        f"이 상황에서 플레이어님은 {h_ko}을 선택했지만, 에이전트는 {a_ko}를 선택했습니다. "
        f"{threat}{reward_desc}"
        f"에이전트는 레인 위협을 피하면서 아이템 수집 경로를 최우선으로 판단했습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 득점은 {h_txt}, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 현재 레인의 위협을 먼저 확인한 뒤 아이템이 있는 레인으로 이동하는 것을 우선 고려해보세요."
    )
