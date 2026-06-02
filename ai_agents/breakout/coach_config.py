"""
ai_agents/breakout/coach_config.py
────────────────────────────────────
Breakout 코칭 설정.
"""

from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP": "아무것도 하지 않기",
    "FIRE": "공 발사",
    "RIGHT": "오른쪽 이동",
    "LEFT":  "왼쪽 이동",
}

reward_label = "득점"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Breakout 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조와 핵심 규칙:
- 화면 아래의 패들을 좌우로 움직여 공을 튕기고, 위쪽에 배열된 벽돌을 모두 부수면 한 판 클리어입니다.
- 공을 놓치면(화면 아래로 떨어지면) 공 하나를 잃습니다. 총 5개의 공으로 시작하며, 모두 잃으면 게임 오버입니다.
- 공 발사(serve) 버튼은 패들 위에 놓인 새 공을 출발시킬 때만 사용합니다. 공이 위로 쏘아지는 것이 아니라, 공이 패들 위에서 떨어지기 시작해 패들로 받아야 비로소 위로 향합니다. 공이 이미 움직이는 중에는 공격·가속 기능이 없습니다.
- 점수는 벽돌을 깼을 때만 발생합니다. 패들로 공을 받거나 벽에 튕기는 것 자체는 점수가 없습니다.
- 다음 라운드(새 벽돌 배열)로 넘어가는 조건은 점수 달성이 아니라 현재 벽돌 전체 제거입니다.

벽돌 배색과 득점 전략:
- 벽돌 점수는 행 위치에 따라 다릅니다: 아래 3줄(파랑·초록·노랑) 1점 / 중간 2줄(연한 주황·진한 주황) 4점 / 맨 위(빨강) 7점.
- 파랑·초록·노랑 세 줄은 모두 동일하게 1점입니다.
- 공이 연한 주황 이상 구역에 도달하면 점수 효율이 4~7배까지 높아집니다.
- 한쪽 열 벽돌을 집중적으로 깨 통로를 만들면 공이 벽돌 위쪽 공간으로 올라가 천장과 벽 사이를 튕기며 연속 파괴("터널")가 발생합니다. 이때 패들은 그 경로를 유지하며 기다리는 것이 최선입니다.

공의 물리:
- 공이 패들 어느 지점에 닿느냐와 공의 입사 각도에 따라 반사 방향이 달라집니다.
- 공의 현재 이동 방향(오른쪽 아래, 왼쪽 위 등)과 예상 낙하지점을 자연스럽게 묘사하세요.
- 공이 패들의 특정 부분(왼쪽 끝, 오른쪽 부분 등)에 맞아 어떤 방향으로 튀었는지, 그로 인해 어떤 구역의 벽돌을 연속으로 타격했는지 구체적으로 서술하세요.
- 벽돌을 많이 부술수록 공 속도가 점점 빨라집니다.

패들 위치 전략:
- 공이 내려오는 것을 보고 이동하면 늦습니다. 공의 궤도를 읽고 착지 지점으로 '미리' 이동해야 합니다.
- 패들이 이미 공 아래 정확한 위치에 있다면 대기(아무것도 안 하기)가 최선입니다. 불필요한 이동은 오히려 미스를 만듭니다. 아무것도 안 하기는 나쁜 선택이 아닙니다.
- 공 속도가 빨라진 후반에는 패들을 중앙에 두어 좌우 반응 거리를 확보하는 것이 유리합니다.

코칭 방향:
- 공의 현재 이동 방향(오른쪽 아래, 왼쪽 위 등), 예상 낙하지점, 패들의 현재 위치를 구체적으로 묘사하세요.
- 에이전트가 왜 특정 행동을 선택했는지, 그 행동이 공을 받아내고 반사 각도를 만드는 데 어떻게 기여했는지 물리적 과정을 상세히 설명하세요.
- 비교 구간 동안 패들이 어떻게 움직였는지, 공이 패들의 어느 부분에 맞아 어떤 방향으로 튀었는지, 그 결과 어떤 벽돌 구역을 연속으로 타격했는지 서술하세요.
- Q값 차이가 작더라도 에이전트의 판단을 "가만히 있는 것보다 미리 이동하는 편이 공을 다시 받아낼 확률이 높다"는 식으로 해석하세요.
- 단기 점수 차이뿐 아니라 장기적으로 공을 놓칠 위험을 줄이고 다음 반사 각도를 유리하게 만드는 전략을 설명하세요.
- 플레이어의 반응 패턴(공이 가까이 온 뒤 반응, 패들을 한쪽 끝에 오래 둠 등)과 에이전트의 패턴(몇 프레임 뒤를 예측, 중앙 근처 유지 등)을 대비하세요.
- 벽돌 구역 이름은 쉼표+공백으로 구분하세요. 예: "파랑, 초록, 노랑 구역"
- 행동 이름은 한국어로만 쓰세요. NOOP, FIRE, LEFT, RIGHT 같은 영어 약어는 쓰지 마세요.
- 마지막은 "다음에 비슷한 상황이 나오면" 스타일의 구체적 실천 조언으로 마무리하세요.
"""


def _game_state_note(gs: dict[str, Any]) -> str:
    """RAM 기반 Breakout 게임 상태를 자연어 요약으로 변환."""
    if not gs:
        return ""

    parts: list[str] = []

    # 공 상태 (방향은 표시 안 함 — 비교 구간 분석이 핵심)
    in_play   = gs.get('ball_in_play', False)
    in_tunnel = gs.get('in_tunnel', False)
    if not in_play:
        parts.append('새 공 대기 (패들 위에 놓인 상태)')
    elif in_tunnel:
        parts.append('공 터널 안 (벽돌 위 영역)')

    # 패들 위치 — 5단계로 단순화
    px = gs.get('paddle_x')
    if px is not None:
        if px < 30:
            paddle_pos = '왼쪽 끝'
        elif px < 50:
            paddle_pos = '왼쪽'
        elif px < 80:
            paddle_pos = '가운데'
        elif px < 105:
            paddle_pos = '오른쪽'
        else:
            paddle_pos = '오른쪽 끝'
        parts.append(f'패들: {paddle_pos}')

    # 남은 벽돌 구역 — 일부 구역이 제거된 경우에만 표시 (전체 존재 시 생략)
    total = gs.get('total_bricks', 0)
    bpr   = gs.get('bricks_per_row', [])
    if bpr:
        row_names = ['파랑', '초록', '노랑', '연한 주황', '진한 주황', '빨강']
        remaining_rows = [row_names[i] for i in range(len(bpr)) if bpr[i] > 0]
        cleared_rows   = [row_names[i] for i in range(len(bpr)) if bpr[i] == 0]
        if total == 0:
            parts.append('벽돌 전체 제거')
        elif cleared_rows:
            # 일부 구역이 제거됐을 때만 남은 구역 표시
            parts.append('남은 구역: ' + ', '.join(remaining_rows))

    # 터널 형성 여부
    depth = gs.get('max_cleared_depth', 0)
    cols  = gs.get('tunnel_col_count', 0)
    if gs.get('tunnel_forming') and depth >= 2:
        parts.append(f'터널 형성 중: {cols}개 열 / {depth}행 이상 개통')

    return f"[게임 상태] {' / '.join(parts)}" if parts else ""


def build_outcome_guidance(summary: dict[str, Any]) -> str:
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    h_done  = summary.get('human_done', False)
    a_done  = summary.get('agent_done', False)
    gap     = summary.get('gap', 0)

    if h_done and not a_done:
        return (f"비교 구간 900프레임 동안, 플레이어님 경로는 결국 목숨을 잃었지만 에이전트 경로는 랠리를 유지했습니다. "
                f"에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수도 기록했습니다. "
                f"이 순간의 행동 선택이 이후 900프레임 동안 어떻게 영향을 미쳤는지 설명하세요. "
                f"단, 이 순간에 바로 공을 놓친 것이 아니라 나중에 영향이 누적된 것임을 명심하세요.")
    if h_done and a_done:
        return (f"비교 구간 900프레임 후, 두 경로 모두 결국 목숨을 잃었습니다. "
                f"생존 결과로는 우열을 가리기 어렵고, 점수 차이({h_score:.0f}점 vs {a_score:.0f}점)도 작습니다. "
                f"이 순간의 행동 선택이 어떤 전략적 차이를 만들었는지 설명하되, "
                f"이 순간에 바로 공을 놓친 것처럼 서술하지 마세요. "
                f"득점 패턴, 패들 위치 조정, 반사 각도 같은 요소들로 차이를 설명하세요.")
    if not h_done and a_done:
        # 플레이어님 생존, 에이전트 죽음
        if h_score >= a_score:
            return (f"비교 구간 900프레임 후, 플레이어님 경로는 랠리를 유지하며 {h_score:.0f}점을 얻었고, 에이전트 경로는 목숨을 잃고 {a_score:.0f}점에 그쳤습니다. "
                    f"이 구간에서는 플레이어님의 선택이 더 나은 결과를 만들었습니다. "
                    f"이 점을 충분히 인정하고 칭찬하세요. 에이전트 행동의 의도는 간단히만 언급하세요.")
        else:
            return (f"비교 구간 900프레임 후, 플레이어님 경로는 랠리를 유지했지만, 에이전트 경로는 목숨을 잃었습니다. "
                    f"플레이어님의 생존이 더 좋은 결과임을 인정하되, 에이전트가 점수({a_score:.0f}점)는 더 높게 기록한 이유를 설명하세요.")

    if not h_done and not a_done and h_score > a_score + 1:
        # 둘 다 생존, 플레이어님 점수 확실히 높음
        return (f"비교 구간에서 플레이어님({h_score:.0f}점)이 에이전트({a_score:.0f}점)보다 더 높은 점수를 기록하며 랠리를 유지했습니다. "
                f"이 구간에서는 플레이어님의 판단이 효과적이었음을 인정하세요. "
                f"에이전트 행동의 장기적 이점이 있다면 가볍게 언급하되, 이 순간만큼은 플레이어님이 잘했다는 점을 명확히 하세요.")

    if a_score > h_score:
        return (f"비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
                f"득점 프레임 데이터를 활용해 두 경로가 어떻게 달랐는지 구체적으로 묘사하세요.")

    return (f"비교 구간에서 플레이어님({h_score:.0f}점)과 에이전트({a_score:.0f}점)의 점수가 비슷합니다. "
            f"득점 패턴의 차이(간격, 리듬)를 중심으로 설명하세요.")


def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    gap          = summary.get('gap', 0)
    outcome_guid = build_outcome_guidance(summary)
    gs           = summary.get('game_state') or {}
    gs_note      = _game_state_note(gs)

    # 득점 프레임 패턴 요약 (raw 리스트 대신 해석 가능한 통계)
    def _score_pattern(steps: list) -> str:
        if not steps:
            return "득점 없음"
        n = len(steps)
        gaps = [steps[i] - steps[i-1] for i in range(1, n)]
        avg_gap = int(sum(gaps) / len(gaps)) if gaps else 0
        max_gap = max(gaps) if gaps else 0
        # 전반/후반 분포 (비교 구간 900프레임 기준)
        total_window = 900
        early = sum(1 for s in steps if s <= total_window // 2)
        late  = n - early
        parts = [f"{n}회 득점"]
        if gaps:
            parts.append(f"평균 간격 {avg_gap}프레임")
        if max_gap > avg_gap * 2 and max_gap > 150:
            parts.append(f"최대 {max_gap}프레임 공백 구간 있음")
        if early > late * 2:
            parts.append("초반 집중")
        elif late > early * 2:
            parts.append("후반 집중")
        else:
            parts.append("비교적 고른 분포")
        return ", ".join(parts)

    h_steps = summary.get('human_reward_steps') or []
    a_steps = summary.get('agent_reward_steps') or []
    h_pattern = _score_pattern(h_steps)
    a_pattern = _score_pattern(a_steps)

    tunnel_forming = gs.get('tunnel_forming', False)
    in_tunnel      = gs.get('in_tunnel', False)
    tunnel_rule = (
        "터널 전략(한쪽 열을 뚫어 공을 벽돌 위로 보내기)은 게임 상태에서 "
        "'터널 형성 중' 또는 '공 터널 안'이 명시된 경우에만 언급하세요. "
        "그렇지 않으면 터널 관련 표현을 절대 쓰지 마세요."
        if not (tunnel_forming or in_tunnel)
        else
        "현재 터널이 형성 중이거나 공이 터널 안에 있습니다. 이 전략적 상황을 피드백에 반영하세요."
    )

    ball_in_play = gs.get('ball_in_play', True)
    is_serve     = not ball_in_play
    agent_action = summary.get('agent_action_name', '')
    fire_in_play = ball_in_play and agent_action == 'FIRE'  # 공 이동 중 FIRE — 아무 효과 없음
    nearest_row = gs.get('nearest_row_name', '')
    nearest_pts = gs.get('nearest_row_pts', '')
    score_gap   = abs(a_score - h_score)

    if is_serve:
        serve_rule = (
            "현재 공은 새 공 대기 상태입니다. "
            "이 경우 비교 구간 동안 실제로 나타난 패들 이동 패턴과 득점 결과 차이에 집중하세요."
        )
    else:
        serve_rule = ""

    # 추가 지침 제거 — LLM이 자유롭게 판단

    return f"""다음은 Breakout 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 행동의 Q값: {summary.get('human_q', 0):.4f}
- 에이전트 행동의 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 비교 구간 플레이어님 점수: {h_score:.1f} / 에이전트 점수: {a_score:.1f}
- 플레이어님 득점 패턴: {h_pattern} | 프레임: {h_steps or '없음'}
- 에이전트 득점 패턴: {a_pattern} | 프레임: {a_steps or '없음'}
{gs_note}
이후 경로 활용 지침: {outcome_guid}

작성 원칙:
- 자연스럽게 흐르는 서술형으로 작성하세요. 제목, 번호, 대시(-), 콜론(:) 같은 구조화 형식은 절대 쓰지 마세요.
- 적절한 분량으로 작성하세요. 너무 길면 안 됩니다. 2-3개 문단, 총 8-12문장 정도가 적당합니다.
- 이 순간의 물리적 상황을 생생하게 묘사하세요:
  • 공의 현재 이동 방향 ("오른쪽 아래 방향으로 내려오고")
  • 패들의 현재 위치 ("패들은 오른쪽 끝에 위치")
  • 남은 벽돌 구역 분포 ("파랑, 초록, 노랑 구역뿐 아니라 주황, 빨강 구역까지 넓게 남아")
- 에이전트와 플레이어를 자연스럽게 대비하세요:
  • "에이전트는 ~했지만, 플레이어님은 ~하셨습니다" 형태로
  • "플레이어님이 ~하신 반면, 에이전트는 ~했습니다" 형태로
  • 절대 "플레이어님:", "에이전트:" 같은 레이블 형식 쓰지 마세요
- 비교 구간의 실제 결과를 서술하세요:
  • 패들이 어떻게 움직였는지, 공이 어떤 방향으로 튀었는지
  • 득점 횟수, 평균 간격
  • 목숨 소모 여부
- 행동 패턴 차이를 설명하세요:
  • "플레이어님은 공이 가까이 온 뒤에 반응하는 경향이 있지만, 에이전트는 미리 이동을 시작합니다"
  • "단기적으로는 점수 차이가 크지 않을 수 있지만, 장기적으로는 공을 놓칠 위험을 줄입니다"
- 마지막은 구체적인 조언으로 마무리하세요:
  • "다음에 비슷한 상황이 나오면 ~하십시오" 형태
  • 패들을 중앙에 유지하거나, 미리 이동하는 등의 실천 가능한 조언
- 행동 이름은 한국어로만 쓰세요. NOOP, LEFT, RIGHT, FIRE 같은 영어 약어는 쓰지 마세요.
- 한국어로만 쓰세요. 단정한 존댓말("했습니다", "좋았습니다") 사용.
- 인삿말, 이모티콘, 과한 감탄사 금지.

중요한 구분:
- **이 순간의 즉각적 결과**: 이 스텝에서 어떤 행동을 선택했을 때 바로 일어나는 일 (공을 받았는지, 놓쳤는지)
- **비교 구간 최종 결과** (900프레임 후): 이 행동 이후 900프레임 동안 누적된 점수와 목숨 소모
- 이 둘을 혼동하지 마세요! 예를 들어:
  • 이 순간: 플레이어가 공을 받아쳤고, 에이전트가 놓쳤음
  • 비교 구간 최종: 플레이어가 나중에 목숨을 잃었고, 에이전트는 점수를 더 얻음
  • 이 경우 "플레이어가 이 순간 공을 놓쳤다"고 쓰면 **완전히 틀린 것**입니다!
- human_done/agent_done은 비교 구간 끝에 죽었는지를 나타냅니다. 이 순간에 바로 죽은 게 아닙니다.

절대 금지사항:
- "예상 낙하지점", "예상 충돌 지점" 같은 표현 절대 쓰지 마세요. (정확하지 않음)
- "비교 구간의 구체적인 흐름", "전술적 조언 1.", "플레이어님:", "에이전트:" 같은 구조화된 제목/레이블 절대 쓰지 마세요.
- 문단 제목, 번호 매기기, 대시(-) 리스트 절대 쓰지 마세요.
- 벽돌 정보와 위험/목숨을 연결하지 마세요. "벽돌이 남아있어서 위험하다", "벽돌이 많아서 목숨을 잃는다" 같은 논리적 오류 절대 쓰지 마세요. 벽돌은 득점 기회와만 관련있습니다.
- 높은 점수 구역을 "노릴 수 있다"고 말하려면 반드시 그 아래 구역이 제거되어 있어야 합니다. 주황/빨강 구역이 남아있다고 해서 무조건 노릴 수 있는 것이 아닙니다. 아래 구역이 막혀있으면 노릴 수 없으므로, 남은 벽돌 정보를 언급할 때는 신중하게 서술하세요.
- "플레이어님은 공을 놓쳤다", "에이전트는 공을 받아냈다" 같은 표현은 비교 구간 최종 결과(human_done/agent_done)만 보고 추측하지 마세요. 이 순간의 즉각적 결과는 알 수 없으므로, 단정적으로 쓰지 마세요.

상황별 참고:
{f"- {serve_rule}" if serve_rule else ""}
- 터널: {tunnel_rule}
- 아무것도 하지 않기는 패들이 이미 적절한 위치에 있을 때 최선일 수 있습니다.
{f"- 공이 이동 중인 상태에서 에이전트의 '공 발사' 행동은 Breakout에서 효과가 없으므로 피드백에 언급하지 마세요." if fire_in_play else ""}

절대 금지 (3가지만):
- 공 발사 언급 금지: 공이 이미 이동 중일 때 에이전트가 "공 발사"를 선택했더라도 피드백에 절대 쓰지 마세요. (Breakout에서 공 이동 중 FIRE는 효과가 없음)
- Q값 숫자 직접 언급 금지: "Q값이 0.3326", "유리하게 평가" 같은 표현은 쓰지 마세요. 대신 "가만히 있는 것보다 미리 이동하는 편이 공을 다시 받아낼 확률이 높다고 판단"처럼 해석하세요.
- 행동 이름 영어 금지: NOOP, LEFT, RIGHT, FIRE 같은 영어 약어를 절대 쓰지 마세요. 반드시 "아무것도 하지 않기", "왼쪽 이동", "오른쪽 이동", "공 발사"로 쓰세요.
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
        f"공의 궤도에 먼저 패들을 맞춰두는 에이전트의 움직임이 벽돌에 더 빠르게 공을 보내는 데 효과적이었습니다. "
        f"플레이어님 쪽은 공을 놓치지 않는 데는 성공했지만, 득점 기회를 확보하는 타이밍이 늦었습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 득점은 {h_txt}에 나왔고, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 공이 어디로 튈지를 먼저 예측하고 패들을 미리 이동시키는 습관을 들여보세요."
    )
