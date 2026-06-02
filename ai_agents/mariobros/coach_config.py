"""ai_agents/mariobros/coach_config.py — Mario Bros 코칭 설정"""
from __future__ import annotations
from typing import Any

_ACTION_KO: dict[str, str] = {
    "NOOP":          "대기",
    "FIRE":          "점프",
    "UP":            "위로 이동",
    "RIGHT":         "오른쪽 이동",
    "LEFT":          "왼쪽 이동",
    "DOWN":          "아래로 이동",
    "UPRIGHT":       "오른쪽 위로 이동",
    "UPLEFT":        "왼쪽 위로 이동",
    "DOWNRIGHT":     "오른쪽 아래로 이동",
    "DOWNLEFT":      "왼쪽 아래로 이동",
    # FIRE(점프)는 항상 위로 솟아오르므로 DOWN+FIRE 조합에서 "아래로"를 제거
    "UPFIRE":        "위로 이동하며 점프",
    "RIGHTFIRE":     "오른쪽으로 점프",
    "LEFTFIRE":      "왼쪽으로 점프",
    "DOWNFIRE":      "점프",           # DOWN 입력이 있어도 점프는 항상 위로
    "UPRIGHTFIRE":   "오른쪽 위로 이동하며 점프",
    "UPLEFTFIRE":    "왼쪽 위로 이동하며 점프",
    "DOWNRIGHTFIRE": "오른쪽으로 점프",  # DOWN 성분 무시 (점프는 항상 위로)
    "DOWNLEFTFIRE":  "왼쪽으로 점프",   # DOWN 성분 무시 (점프는 항상 위로)
}

reward_label = "득점"


def action_name_ko(name: str) -> str:
    return _ACTION_KO.get(name, name)


SYSTEM_PROMPT = """당신은 Mario Bros 플레이어님에게 1:1 코칭을 제공하는 게임 전략 코치입니다.

게임 구조와 핵심 규칙:
- 화면 안의 여러 층 발판 위에서 파이프를 통해 쏟아지는 적들을 처치하는 아케이드 액션 게임입니다.
- 적을 처치하는 방법은 두 단계입니다: ①적이 올라서 있는 발판의 바로 아래에서 점프해 그 발판을 쳐서 적을 뒤집어 기절시키기 → ②기절한 적에게 달려가 발로 차서 처치하기. 이 두 단계를 모두 해야 득점입니다.
- 기절 상태는 짧은 시간만 유지됩니다. 차러 가지 않으면 적이 다시 일어나며, 다시 일어난 적은 더 빠르게 움직입니다.
- POW 블록: 화면 중앙에 있는 POW 블록을 치면 화면 전체를 진동시켜 모든 적을 한 번에 기절시킬 수 있습니다. 3번 사용하면 사라지는 희귀 자원이므로 여러 적이 한꺼번에 있을 때 써야 가장 효율적입니다.

적의 종류:
- 셸크리퍼(거북): 발판을 천천히 걷는 기본 적. 기절 시간이 길어 비교적 처리하기 쉽습니다.
- 사이드스테퍼(게): 좌우로 빠르게 움직이며 방향을 자주 바꿉니다. 한 번 기절시킨 뒤 일어나면 이전보다 더 빠르게 움직입니다.
- 파이터 플라이(파리): 발판 위아래를 날아다닙니다. 땅에 잠깐 내려앉을 때만 기절시킬 수 있어 타이밍이 중요합니다.
- 라운드가 올라갈수록 적의 수와 속도가 증가하고, 여러 종류가 동시에 등장합니다.

연쇄 처치 보너스:
- 한 번의 발판 치기로 기절한 여러 적을 빠르게 연속으로 차면 보너스 점수가 누적됩니다.
- 기절 타이머를 의식하며 빠른 순서로 처리하는 것이 고득점의 핵심입니다.

이동과 점프 전략:
- 점프 액션은 항상 위쪽으로 솟아오릅니다. 방향 입력(왼쪽/오른쪽)은 점프하는 동안의 수평 이동 방향을 결정할 뿐입니다.
- "아래 방향 + 점프" 조합도 실제로는 위로 점프합니다. DOWN 입력이 점프 방향을 바꾸지 않습니다.
- 이동과 점프를 동시에 처리하는 복합 액션을 사용하면 더 빠르게 위치를 잡을 수 있습니다.
- 발판 아래로 이동해 점프하는 타이밍이 맞아야 적을 기절시킬 수 있으므로, 목표 발판의 바로 아래에서 점프하는 위치 선정이 핵심입니다.
- 여러 적이 동시에 있을 때는 가장 빠르게 이동하거나 플레이어 쪽으로 내려오는 적부터 우선 처리해야 합니다.

코칭 방향:
- "에이전트가 왜 이 타이밍에 이 방향으로 이동하며 점프했는가"를 현재 적의 위치, 기절 타이머, 가장 위협적인 적의 경로 기준으로 설명하세요.
- 행동 이름은 한국어 이름만 사용하세요. NOOP, FIRE, RIGHTFIRE, UPLEFTFIRE 같은 영어 약어는 절대 쓰지 마세요.
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
                f"이동 경로와 점프 타이밍 차이가 생존에 어떤 영향을 미쳤는지 설명하세요.")
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
                f"득점 프레임 데이터를 활용해 두 경로의 차이를 묘사하세요.")
    return (f"이후 약 30초 비교 구간에서는 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
            f"대신 이 순간의 Q값 차이({gap:.4f})의 전략적 의미에 집중하세요.")


def build_user_prompt(summary: dict[str, Any]) -> str:
    h_score      = summary.get('human_score_delta', 0)
    a_score      = summary.get('agent_score_delta', 0)
    outcome_guid = build_outcome_guidance(summary)

    return f"""다음은 Mario Bros 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {action_name_ko(summary['human_action_name'])}
- 에이전트 행동: {action_name_ko(summary['agent_action_name'])}
- 플레이어님 Q값: {summary.get('human_q', 0):.4f} / 에이전트 Q값: {summary.get('agent_q', 0):.4f} / 가치 차이: {summary.get('gap', 0):.4f}
- 이후 30초 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 30초 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 첫 득점 시점: {summary.get('human_first_reward_step', '득점 없음')}
- 에이전트 쪽 첫 득점 시점: {summary.get('agent_first_reward_step', '득점 없음')}
- 플레이어님 쪽 득점 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 득점 프레임들: {summary.get('agent_reward_steps') or '없음'}

이후 경로 활용 지침: {outcome_guid}

작성 방식:
- 문단 수는 2~3개로 자유롭게 구성하세요. 각 문단은 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 행동 이름은 반드시 한국어로만 표현하세요. NOOP, FIRE, RIGHTFIRE, UPLEFTFIRE 같은 영어 약어는 절대 쓰지 마세요.
- 첫 번째 문단: 플레이어님 행동과 에이전트 행동을 대비하되, 핵심은 "에이전트가 왜 그 방향으로 이동하며 점프했는가"입니다. 적의 위치, 기절 타이머, 발판 위치를 기준으로 설명하세요. "이후 경로 활용 지침"에 따라 이후 전개도 포함하세요.
- 중간 문단(선택): 여러 적이 동시에 있을 때의 우선순위 판단, 기절 타이머를 의식한 연쇄 처치 전략, POW 블록 활용 타이밍 같은 게임 메커니즘 관점에서 에이전트의 선택이 왜 더 효율적이었는지 설명하세요.
- 마지막 문단: "다음에 이런 상황이라면" 뉘앙스로 시작해 즉시 실천 가능한 조언으로 마무리하세요.
- 점수를 언급할 때는 "이 구간에서" 또는 "비교 구간에서"라고 명시하세요.
- 숫자 나열보다 "사이드스테퍼가 이미 기절한 뒤 일어나기 직전이어서", "위층 발판에 적이 몰려 있어 한 번의 타격으로 여러 마리를 기절시킬 수 있는 타이밍이었기 때문에" 같은 구체적 표현을 사용하세요.
- 한국어로만 쓰세요.
- "했습니다", "좋았습니다", "유리했습니다"처럼 단정한 존댓말을 사용하세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
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
        f"에이전트는 이동과 점프를 정밀하게 조합해 적을 처치하면서 "
        f"득점 흐름을 앞당겼습니다.\n\n"
        f"비교 구간에서 플레이어님 쪽 득점은 {h_txt}에, 에이전트 쪽 득점은 {a_txt}에 이어졌습니다. "
        f"다음에 이런 상황이라면, 적에게 접근하기 전 점프 타이밍을 미리 계산하고 이동 방향과 함께 활용해보세요."
    )
