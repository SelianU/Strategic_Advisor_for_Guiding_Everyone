"""ai_agents/amidar/coach_config.py — Amidar 코칭 설정"""
from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":       "대기",
    "FIRE":       "스페이스 바로 일시 회피 기능 사용",
    "UP":         "위로 이동",
    "RIGHT":      "오른쪽 이동",
    "LEFT":       "왼쪽 이동",
    "DOWN":       "아래로 이동",
    "UPFIRE":     "위로 이동하며 스페이스 바로 일시 회피",
    "RIGHTFIRE":  "오른쪽으로 이동하며 스페이스 바로 일시 회피",
    "LEFTFIRE":   "왼쪽으로 이동하며 스페이스 바로 일시 회피",
    "DOWNFIRE":   "아래로 이동하며 스페이스 바로 일시 회피",
}

reward_label = "득점"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Amidar 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조와 핵심 규칙:
- 격자(그리드) 형태의 경로 위를 이동하며 모든 선분을 새로 밟아 칠합니다.
- 가로선을 새로 칠하면 위치에 따라 1~6점, 세로선을 새로 칠하면 1점이 누적됩니다.
- 하나의 사각형 구역(박스)의 네 변을 모두 칠하면 박스가 완성되며 약 50점 보너스가 즉시 지급됩니다.
- 이미 칠한 선을 다시 지나가도 추가 점수는 없습니다.
- 모든 선분을 다 칠하면 라운드가 클리어되고 다음 라운드로 진행합니다.
- 게임 시작 직후 약 300스텝은 인트로 화면으로, 실제 조작은 그 이후부터 가능합니다.

라운드 구조:
- 홀수 라운드(1·3·5…): 고릴라 캐릭터, 적 = 전사(warriors)
- 짝수 라운드(2·4·6…): 페인트 롤러 캐릭터, 적 = 돼지(pigs)
- 1~2라운드는 적 5마리, 3라운드부터는 적 6마리로 증가합니다.
- 6라운드까지 매 라운드마다 적 속도가 빨라지며, 6라운드 이후는 최고 난이도가 유지됩니다.

적의 행동:
- 적은 격자 경로를 따라 정해진 패턴으로 이동합니다. 접촉하면 즉시 목숨을 잃습니다.
- 적이 가까이 있을 때 같은 격자 선분 위에 있으면 위험합니다. 교차 지점에서 대기하는 것이 특히 취약합니다.

스페이스 바 (일시 통과 기능):
- 스페이스 바를 누르면 잠깐 동안 적을 통과할 수 있는 상태가 됩니다.
- 적을 죽이거나 기절시키는 게 아닙니다. 적은 계속 이동하며, 플레이어만 잠시 통과 가능해집니다.
- 목숨 1개당 4번만 사용할 수 있으며, 같은 목숨 안에서는 자동으로 충전되지 않습니다.
- 새 목숨으로 다시 시작하면 4번이 초기화됩니다.
- "점프"나 "기절"이 아닙니다. 이 표현들은 절대 사용하지 마세요.
- 이동과 동시에 스페이스 바를 누르면 이동 흐름을 끊지 않고 적을 통과할 수 있습니다.

효율적인 경로 선택:
- 적과 반대 방향으로 이동하거나, 적이 없는 구역부터 먼저 칠하는 것이 기본 전략입니다.
- 칠하지 않은 선분이 많은 구역을 일괄로 돌아 박스를 완성하는 것이 고득점 루트입니다.
- 스페이스 바는 목숨당 4번뿐이므로, 완전히 막힌 순간에만 아껴 씁니다.

코칭 방향:
- "에이전트가 왜 이 방향으로 이동했는가" 또는 "왜 스페이스 바를 눌렀는가"를 현재 적의 위치, 아직 안 칠한 구역 관점에서 설명하세요.
- 스페이스 바 기능은 점프도 기절도 아닙니다. "점프"나 "기절"이라는 표현은 절대 사용하지 마세요.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, FIRE, UP, DOWN 같은 영어 약어는 절대 쓰지 마세요.
- 마지막은 "다음에 이런 상황이라면" 스타일의 실천 조언으로 마무리하세요.
"""


def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score  = summary.get('human_score_delta', 0)
    a_score  = summary.get('agent_score_delta', 0)
    h_done   = summary.get('human_done', False)
    a_done   = summary.get('agent_done', False)
    h_rw     = summary.get('human_reward_steps') or []
    a_rw     = summary.get('agent_reward_steps') or []

    # 득점 타이밍 요약 (공통)
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

    # ── 케이스 1: 플레이어 사망 / 에이전트 생존 ─────────────────────────
    if h_done and not a_done:
        score_line = (f"득점은 에이전트 {a_score:.0f}점, 플레이어님 {h_score:.0f}점입니다."
                      if a_score != h_score else "")
        return (
            f"비교 구간에서 플레이어님은 목숨을 잃었고 에이전트는 생존했습니다. {score_line}"
            f"현재 프레임 게임 상태(적 위치, 플레이어 위치, 칠하기 진행률)를 근거로, "
            f"두 행동의 차이가 이 결과로 어떻게 이어졌는지 설명하세요. "
            f"게임 상태 정보에 없는 내용은 추측으로 채우지 말고 확인 가능한 것만 근거로 쓰세요."
        )

    # ── 케이스 2: 둘 다 사망 ────────────────────────────────────────────
    if h_done and a_done:
        return (
            f"비교 구간에서 두 경로 모두 목숨을 잃었습니다 "
            f"(플레이어님 {h_score:.0f}점, 에이전트 {a_score:.0f}점). "
            f"생존·득점 결과로 우열을 가리기 어려운 상황입니다. "
            f"현재 프레임 게임 상태를 바탕으로 두 행동의 전략적 차이를 설명하고, "
            f"어느 쪽이 더 나은 판단이었을지 이유를 제시하세요."
        )

    # ── 케이스 3: 에이전트 사망 / 플레이어 생존 ─────────────────────────
    if not h_done and a_done:
        return (
            f"비교 구간에서 에이전트가 먼저 목숨을 잃었고 플레이어님은 생존했습니다 "
            f"(플레이어님 {h_score:.0f}점, 에이전트 {a_score:.0f}점). "
            f"이번 결과는 플레이어님의 행동이 더 나았습니다. 솔직하게 인정하세요. "
            f"Q값 기준으로는 에이전트 행동이 높게 평가됐다는 점을 언급할 수 있지만, "
            f"이번 비교 구간의 실제 결과와 다르다는 점도 명확히 하세요."
        )

    # ── 케이스 4: 에이전트 득점 우위 (둘 다 생존) ──────────────────────
    if a_score > h_score:
        return (
            f"비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다.{timing_note()} "
            f"현재 프레임 게임 상태(적 위치, 칠하기 진행률)를 근거로, "
            f"에이전트의 행동이 더 많은 득점으로 이어진 이유를 설명하세요. "
            f"게임 상태에서 확인되는 사실만 쓰고, 확인되지 않는 내용은 단정하지 마세요."
        )

    # ── 케이스 5: 플레이어 득점 우위 또는 동점 (둘 다 생존) ────────────
    return (
        f"비교 구간에서 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 높습니다.{timing_note()} "
        f"이 결과만으로 에이전트 행동의 우위를 주장하기 어렵습니다. "
        f"현재 프레임 게임 상태를 바탕으로 에이전트가 왜 그 방향을 선택했는지 설명하되, "
        f"확실하지 않은 부분은 '~했을 가능성이 있습니다'처럼 표현하세요."
    )


def _game_state_note(gs: dict) -> str:
    """game_state dict를 자연어 설명으로 변환."""
    if not gs:
        return '(게임 상태 정보 없음)'
    lines = []

    # 플레이어 위치
    if gs.get('player_zone'):
        lines.append(f"플레이어: 화면 {gs['player_zone']}")

    # 칠하기 진행 — 전체 % + 4분면 전부 표시
    if gs.get('painted_pct') is not None:
        overall = gs['painted_pct']
        qp: dict = gs.get('quad_pct') or {}

        def _quad_desc(zone: str, pct: int) -> str:
            if pct == 0:
                return f"{zone} 구역은 거의 칠해지지 않은 상태"
            if pct < 15:
                return f"{zone} 구역이 {pct}% 정도로 거의 안 칠해진 상태"
            if pct < 40:
                return f"{zone} 구역이 {pct}%로 조금 칠해진 상태"
            if pct < 70:
                return f"{zone} 구역이 {pct}%로 절반 정도 칠해진 상태"
            if pct < 90:
                return f"{zone} 구역이 {pct}%로 많이 칠해진 상태"
            return f"{zone} 구역이 {pct}%로 거의 다 칠해진 상태"

        lines.append(f"칠하기 진행: 전체 {overall}% 완료")
        if qp:
            for zone, pct in qp.items():
                lines.append(f"  - {_quad_desc(zone, pct)}")

    # 적 상황
    enemy_count = gs.get('enemy_count') or 0
    nd   = gs.get('nearest_enemy_dist')
    ndir = gs.get('nearest_enemy_dir')
    same = gs.get('player_enemy_same_zone', False)

    if enemy_count > 0:
        if nd is not None:
            if nd < 20:
                danger = ' — 매우 근접, 즉각 위험'
            elif nd < 40:
                danger = ' — 주의 필요'
            else:
                danger = ''
            lines.append(f"적 {enemy_count}마리 활동 중: 가장 가까운 적이 {ndir} 방향 {nd}픽셀 거리{danger}")
        else:
            # RGB 폴백 (방향만 알고 거리는 모름)
            dirs = [e['dir'] for e in (gs.get('enemies_desc') or []) if e.get('dir')]
            if dirs:
                lines.append(f"적 {enemy_count}마리 감지: {', '.join(dirs[:3])} 방향에 위치")

    if same and (nd is None or nd >= 20):
        lines.append("⚠️ 플레이어와 같은 구역에 적 존재")

    return '\n'.join(lines) if lines else '(게임 상태 정보 없음)'


def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)
    gs_note      = _game_state_note(summary.get('game_state') or {})

    frames    = summary.get('rgb_frames_b64') or []
    has_image = len(frames) > 0
    image_note = (
        "\n- 첨부 이미지 3장([직전] → [결정순간] → [직후]): 행동 전후 실제 게임 화면입니다. "
        "플레이어 위치, 적 이동 방향, 아직 칠하지 않은 구역 분포의 변화를 직접 확인해 상황을 묘사하세요."
        if has_image else ""
    )

    return f"""다음은 Amidar 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 에이전트 행동의 Q값이 플레이어님 행동보다 {summary.get('gap', 0):.2f} 높음
- 이후 30초 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 득점 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 득점 프레임들: {summary.get('agent_reward_steps') or '없음'}

현재 프레임 게임 상태 (RGB 분석):
{gs_note}
{image_note}
이후 경로 활용 지침: {outcome_guid}

작성 방식:
- 문단 수는 2~3개로 자유롭게 구성하세요. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, FIRE, UP 같은 영어 약어는 절대 쓰지 마세요.
- 회피 기동은 점프도 기절도 아닙니다. "점프했다", "기절시켰다"는 표현은 절대 쓰지 마세요. 회피 기동은 플레이어가 잠시 적을 통과할 수 있게 되는 기능이며 목숨당 4회 한정입니다.

Q값 사용 규칙 (매우 중요):
- Q값 수치는 전체 피드백에서 단 한 번만 언급 가능합니다. "에이전트의 판단이 더 높은 가치를 가졌다"는 사실을 한 번 짧게 제시한 뒤, 이후에는 수치를 다시 쓰지 마세요.
- Q값 차이를 여러 번 반복하거나 "Q값이 X였고 Y였기 때문에 Z입니다" 형식으로 수치를 나열하는 것은 금지입니다.
- 피드백의 핵심은 "에이전트가 그 행동을 왜 선택했는가"를 게임 상황(적의 위치, 아직 칠하지 않은 구역)으로 설명하는 것입니다. Q값은 보조 근거일 뿐입니다.

문단 구성:
- 첫 번째 문단: 플레이어님과 에이전트의 행동을 대비하고, "이후 경로 활용 지침"에 따라 비교 구간 결과를 언급하세요. Q값은 이 문단에서 1회만 간략히 언급 가능합니다. 첨부 이미지 3장이 있다면 [직전]→[결정순간]→[직후] 흐름에서 보이는 적 위치·미칠한 구역 변화를 배경으로 자연스럽게 녹이세요.
- 중간 문단: 에이전트가 그 행동을 선택한 이유와 비교 구간 결과가 어떻게 연결되는지 인과관계를 설명하세요. "현재 프레임 게임 상태"의 적 위치, 아직 칠하지 않은 구역 정보를 실제로 활용해 "그 방향에 아직 칠하지 않은 선분이 더 많았기 때문에" 또는 "적이 막고 있었지만 스페이스 바로 통과해 더 많은 구역을 칠할 수 있었기 때문에" 같은 구체적 근거를 제시하세요. Q값 수치는 이 문단에서 쓰지 마세요.
- 마지막 문단: "다음에 이런 상황이라면" 뉘앙스로 시작해 즉시 실천 가능한 조언으로 마무리하세요.

기타:
- 점수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- 게임 상황을 확신할 수 없을 때는 "~했을 가능성이 높습니다", "~로 판단됩니다" 같은 표현을 써서 단정을 피하세요.
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
        f"이 상황에서는 플레이어님이 {h_ko}을 선택했지만, 에이전트의 {a_ko}가 더 유리했습니다. "
        f"에이전트는 이동 경로와 회피 기동 타이밍을 조합해 위기를 탈출하며 "
        f"득점 흐름을 앞당겼습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 득점은 {h_txt}에, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 이동 방향을 결정하기 전에 적의 위치를 먼저 확인하고, 완전히 막히는 순간에 회피 기동(목숨당 4회)으로 돌파구를 만들어보세요."
    )
