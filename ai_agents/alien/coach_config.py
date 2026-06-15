"""ai_agents/alien/coach_config.py — Alien 코칭 설정"""
from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":          "대기",
    "FIRE":          "화염 방사기 사용",
    "UP":            "위로 이동",
    "RIGHT":         "오른쪽 이동",
    "LEFT":          "왼쪽 이동",
    "DOWN":          "아래로 이동",
    "UPRIGHT":       "오른쪽 위로 이동",
    "UPLEFT":        "왼쪽 위로 이동",
    "DOWNRIGHT":     "오른쪽 아래로 이동",
    "DOWNLEFT":      "왼쪽 아래로 이동",
    "UPFIRE":        "위로 이동하며 화염 방사",
    "RIGHTFIRE":     "오른쪽으로 이동하며 화염 방사",
    "LEFTFIRE":      "왼쪽으로 이동하며 화염 방사",
    "DOWNFIRE":      "아래로 이동하며 화염 방사",
    "UPRIGHTFIRE":   "오른쪽 위로 이동하며 화염 방사",
    "UPLEFTFIRE":    "왼쪽 위로 이동하며 화염 방사",
    "DOWNRIGHTFIRE": "오른쪽 아래로 이동하며 화염 방사",
    "DOWNLEFTFIRE":  "왼쪽 아래로 이동하며 화염 방사",
}

reward_label = "득점"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Alien 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조와 핵심 규칙:
- 미로형 우주선 내부를 이동하며 에그(Egg)를 모두 제거하면 다음 라운드로 진행합니다.
- 에그 위를 지나가면 자동으로 제거되며 +10점이 지급됩니다. 점수가 아니라 에그 전체 제거가 라운드 클리어 조건입니다.
- 외계인에게 직접 닿으면 즉시 목숨을 잃습니다. 목숨은 3개입니다.

에그(Egg):
- 미로 곳곳에 배치된 흰 점 형태의 목표물입니다.
- 위를 지나가면 제거되며 +10점을 얻습니다.
- 현재 미로의 에그를 모두 제거해야 다음 라운드로 진행합니다.

펄사(Pulsar):
- 한 미로당 최대 3개 등장하는 파워업 아이템입니다.
- 획득 시 +10점, 외계인이 일시적으로 약화됩니다.
- 약화된 외계인에게 닿으면 외계인을 제거할 수 있습니다. (+500점)
- 펄사 효과가 끝나면 외계인은 다시 위험한 상태가 됩니다.

외계인 처치:
- 펄사 효과 중에만 외계인에게 닿아서 제거할 수 있습니다. (+500점)
- 평소에 외계인에게 닿으면 목숨을 잃습니다.

프라이즈(Prize):
- 미로 중앙에 등장하는 노란색 화살/로켓 모양의 보너스 아이템입니다.
- 한 미로당 최대 2개 등장합니다.
- 획득 시 +500점. 라운드 클리어 조건은 아니지만 고득점에 중요합니다.

화염 방사기 (스페이스 바):
- 스페이스 바로 화염 방사기를 사용합니다.
- 외계인을 직접 제거하는 공격이 아닙니다. 가까이 온 외계인을 밀어내거나 잠깐 멈추게 해 도망갈 시간을 버는 방어 수단입니다.
- 이동과 동시에 사용할 수 있습니다. 좁은 통로에서 외계인과 마주쳤을 때 탈출에 유용합니다.
- "총을 쐈다", "발사했다", "공격했다"는 표현은 절대 사용하지 마세요. 이 기능은 방어입니다.

미로 구조:
- 미로 중앙의 좌우 통로는 서로 연결되어 있습니다. 왼쪽 끝으로 나가면 오른쪽 끝에서 등장하고, 오른쪽 끝으로 나가면 왼쪽 끝에서 등장합니다.
- 외계인에게 쫓길 때 이 통로를 이용해 위치를 뒤바꾸는 것이 유효한 탈출 전략입니다.

보너스 라운드 (다크 모드):
- 미로 클리어 후 배경이 검게 변하는 보너스 라운드가 등장합니다.
- 여러 색깔의 외계인들이 화면을 자유롭게 돌아다리며, 화면 위쪽에 프라이즈 우주선이 날아다닙니다.
- 프라이즈 우주선에 닿으면 고득점 보너스를 얻을 수 있습니다.
- 외계인이 매우 많으므로 이동 경로를 신중하게 잡아야 합니다. 펄사가 있다면 약화 효과를 활용해 틈을 노리세요.
- 게임 상태 데이터에 "다크 모드"로 표시되면 이 보너스 라운드 상황입니다.

라운드 진행:
- 라운드가 올라갈수록 외계인 속도가 빨라져 난이도가 증가합니다.

코칭 방향:
- "에이전트가 왜 이 방향으로 이동했는가", "왜 화염 방사기를 사용했는가"를 에그 위치, 외계인 위치, 펄사 상태 기준으로 설명하세요.
- 화염 방사기는 방어·생존 수단입니다. "발사했다", "쐈다", "공격했다"는 표현은 절대 사용하지 마세요.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, FIRE, UPFIRE 같은 영어 약어는 절대 쓰지 마세요.
- Q값 수치는 전체 피드백에서 단 한 번만 언급 가능합니다.
- 마지막은 "다음에 이런 상황이라면" 스타일의 실천 조언으로 마무리하세요.
"""


def _game_state_note(gs: dict[str, Any]) -> str:
    """RAM 기반 Alien 게임 상태를 자연어 요약으로 변환."""
    if not gs:
        return ""

    parts: list[str] = []

    # 다크 모드 여부
    if gs.get('game_mode') == 'dark':
        parts.append("다크 모드 진행 중")

    # 외계인 근접 상황
    alien_count  = gs.get('alien_count', 0)
    nearest_dist = gs.get('nearest_alien_dist')
    nearest_dir  = gs.get('nearest_alien_dir')
    if alien_count > 0 and nearest_dist is not None:
        if nearest_dist < 40:
            danger = '위험 근접'
        elif nearest_dist < 80:
            danger = '주의'
        else:
            danger = '여유'
        parts.append(
            f"외계인 {alien_count}마리 활성 / "
            f"최근접 {nearest_dir} {nearest_dist}px ({danger})"
        )
    else:
        parts.append("외계인 없음")

    # 외계인 약화 / 펄사 상태
    if gs.get('alien_vulnerable'):
        parts.append("외계인 약화 중 — 처치 가능")
    elif gs.get('pulsar_active'):
        pulsar_pos = gs.get('pulsar_position', '')
        parts.append(f"펄사 {pulsar_pos}에 있음 (미획득)")

    # 남은 에그
    eggs = gs.get('eggs_remaining')
    if eggs is not None:
        parts.append(f"현재 미로 에그 {eggs}개 남음")

    # 잔여 목숨
    lives = gs.get('lives')
    if lives is not None:
        parts.append(f"잔여 목숨 {lives}개")

    return f"[게임 상태] {' / '.join(parts)}" if parts else ""


def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    h_rw    = summary.get('human_reward_steps') or []
    a_rw    = summary.get('agent_reward_steps') or []

    def timing_note():
        a_first = a_rw[0] if a_rw else None
        h_first = h_rw[0] if h_rw else None
        if a_first and h_first:
            if a_first < h_first:
                return f" 에이전트가 먼저({a_first}프레임) 득점을 시작했고, 플레이어님은 {h_first}프레임부터 득점했습니다."
            else:
                return f" 플레이어님이 먼저({h_first}프레임) 득점했지만, 에이전트는 이후 더 지속적으로 득점해 역전했습니다."
        elif a_first:
            return f" 에이전트만 {a_first}프레임부터 득점했고, 플레이어님은 이 구간 동안 득점하지 못했습니다."
        elif h_first:
            return f" 플레이어님만 {h_first}프레임부터 득점했고, 에이전트는 이 구간 동안 득점하지 못했습니다."
        return ""

    if h_done and not a_done:
        score_line = (f"득점은 에이전트 {a_score:.0f}점, 플레이어님 {h_score:.0f}점입니다."
                      if a_score != h_score else "")
        return (
            f"비교 구간에서 플레이어님은 목숨을 잃었고 에이전트는 생존했습니다. {score_line} "
            f"두 행동의 차이가 이 결과로 어떻게 이어졌는지 설명하세요. "
            f"확인되지 않는 내용은 추측으로 채우지 마세요."
        )
    if h_done and a_done:
        return (
            f"비교 구간에서 두 경로 모두 목숨을 잃었습니다 "
            f"(플레이어님 {h_score:.0f}점, 에이전트 {a_score:.0f}점). "
            f"두 행동의 전략적 차이를 설명하고 어느 쪽이 더 나은 판단이었는지 이유를 제시하세요."
        )
    if not h_done and a_done:
        return (
            f"비교 구간에서 에이전트가 먼저 목숨을 잃었고 플레이어님은 생존했습니다 "
            f"(플레이어님 {h_score:.0f}점, 에이전트 {a_score:.0f}점). "
            f"이번 결과는 플레이어님의 행동이 더 나았습니다. 솔직하게 인정하세요. "
            f"Q값 기준으로는 에이전트 행동이 높게 평가됐다는 점은 언급할 수 있지만, 실제 결과와 다르다는 점도 명확히 하세요."
        )
    if a_score > h_score:
        return (
            f"비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다.{timing_note()} "
            f"에이전트의 행동이 더 많은 득점으로 이어진 이유를 설명하세요. "
            f"확인되지 않는 내용은 단정하지 마세요."
        )
    return (
        f"비교 구간에서 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 높습니다.{timing_note()} "
        f"에이전트 행동의 우위를 주장하기 어렵습니다. "
        f"에이전트가 왜 그 방향을 선택했는지 설명하되, 확실하지 않은 부분은 '~했을 가능성이 있습니다'처럼 표현하세요."
    )


def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)
    gs_note      = _game_state_note(summary.get('game_state') or {})

    frames    = summary.get('rgb_frames_b64') or []
    has_image = len(frames) > 0
    image_note = (
        "\n- 첨부 이미지 3장([직전] → [결정순간] → [직후]): 행동 전후 실제 게임 화면입니다. "
        "외계인 위치·이동 방향, 에그 분포, 펄사 상태, 플레이어 위치의 변화를 직접 확인해 상황을 묘사하세요."
        if has_image else ""
    )

    return f"""다음은 Alien 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 에이전트 행동의 Q값이 플레이어님 행동보다 {summary.get('gap', 0):.2f} 높음
- 이후 30초 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 득점 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 득점 프레임들: {summary.get('agent_reward_steps') or '없음'}
{gs_note}{image_note}
이후 경로 활용 지침: {outcome_guid}

작성 방식:
- 문단 수는 2~3개로 자유롭게 구성하세요. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, FIRE, UPFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 화염 방사기는 방어 수단입니다. "발사했다", "쐈다"는 표현은 절대 쓰지 마세요.
- Q값 수치는 전체 피드백에서 단 한 번만 언급하세요.
- 첫 번째 문단: 플레이어님과 에이전트의 행동을 대비하고, "이후 경로 활용 지침"에 따라 비교 구간 결과를 언급하세요. 첨부 이미지 3장이 있다면 [직전]→[결정순간]→[직후] 흐름에서 보이는 외계인 위치·에그 분포·펄사 상태를 배경으로 자연스럽게 녹이세요.
- 중간 문단: 에이전트가 그 행동을 선택한 이유를 외계인 위치, 에그 분포, 펄사 상태 관점에서 설명하세요.
- 마지막 문단: "다음에 이런 상황이라면" 뉘앙스로 시작해 즉시 실천 가능한 조언으로 마무리하세요.
- "했습니다", "좋았습니다" 같은 단정한 존댓말 사용. 인삿말, 이모티콘, 과한 감탄사 금지.
- 한국어로만 쓰세요.
"""


def build_fallback_feedback(summary: dict[str, Any]) -> str:
    h_ko    = action_name_ko(summary["human_action_name"])
    a_ko    = action_name_ko(summary["agent_action_name"])
    h_steps = summary.get("human_reward_steps", [])
    a_steps = summary.get("agent_reward_steps", [])
    h_txt   = ", ".join(f"{s}프레임" for s in h_steps[:5]) if h_steps else "득점이 없었습니다"
    a_txt   = ", ".join(f"{s}프레임" for s in a_steps[:5]) if a_steps else "득점이 없었습니다"
    return (
        f"이 상황에서 플레이어님은 {h_ko}을 선택했지만, 에이전트는 {a_ko}를 선택했습니다. "
        f"에이전트는 외계인 위치와 에그 분포를 고려해 더 안전한 경로를 택했습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 득점은 {h_txt}에, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 외계인이 없는 방향의 에그부터 먼저 제거하고, "
        f"펄사를 먹은 직후를 외계인 처치 기회로 활용해 보세요."
    )
