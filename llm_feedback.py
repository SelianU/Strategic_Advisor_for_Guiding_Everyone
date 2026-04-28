import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
FALLBACK_MODELS = [
    # ── 1순위: 대형 모델 (429 가능하지만 품질 우수) ──────────────────
    "nousresearch/hermes-3-llama-3.1-405b:free",  # Llama 3.1 405B 기반
    "google/gemma-4-31b-it:free",                 # Gemma 4 31B
    # ── 2순위: 실제 작동 확인된 중형 모델 ────────────────────────────
    "openai/gpt-oss-120b:free",                   # 120B, 안정적
    "openai/gpt-oss-20b:free",                    # 20B, 안정적
    "google/gemma-3-27b-it:free",                 # 27B, 안정적
    "nvidia/nemotron-3-super-120b-a12b:free",     # 120B Nvidia
    # ── 3순위: 최후 보루 ──────────────────────────────────────────────
    "google/gemma-3-12b-it:free",                 # 12B, 거의 항상 응답
]
MAX_LLM_ATTEMPTS = 2
RETRY_DELAYS = [2.0, 4.0]

ACTION_NAME_KO = {
    # Space Invaders
    "NOOP": "아무것도 하지 않기",
    "FIRE": "발사",
    "LEFT": "왼쪽 이동",
    "RIGHT": "오른쪽 이동",
    "LEFTFIRE": "왼쪽으로 이동하며 발사",
    "RIGHTFIRE": "오른쪽으로 이동하며 발사",
}

BO_ACTION_NAME_KO = {
    "NOOP": "아무것도 하지 않기",
    "FIRE": "공 발사",
    "RIGHT": "오른쪽 이동",
    "LEFT": "왼쪽 이동",
}

SPACE_INVADERS_CONTEXT = """당신은 Space Invaders 플레이어님에게 1:1 코칭을 해주는 게임 코치입니다.

역할:
- 플레이어님이 한 행동과 AI 에이전트가 선택한 행동을 비교하고, 에이전트의 선택이 왜 더 좋은 판단이었는지 설명합니다.
- 그 상황에서 어떻게 행동해야 했는지 플레이어님이 다음 판에 바로 써먹을 수 있는 조언을 줍니다.
- 분석 보고서가 아니라, 옆에서 직접 말해주는 코치처럼 자연스럽게 설명하세요.

게임 정보:
- 플레이어님은 화면 아래에서 좌우로 움직이며 위쪽 적들을 쏴 맞춥니다.
- 적이 줄수록 이동 속도가 빨라집니다. 방패는 탄을 막아주며 점차 소모됩니다.
- 모든 적을 처치하면 스테이지가 클리어되고 다음 스테이지로 넘어갑니다(게임오버가 아닙니다).
- 주요 행동: NOOP(대기), FIRE(발사), LEFT(왼쪽), RIGHT(오른쪽), LEFTFIRE(왼쪽+발사), RIGHTFIRE(오른쪽+발사)

코칭 원칙:
- 플레이어님의 행동과 에이전트 행동을 반드시 대비해서 설명하세요.
- "에이전트가 왜 더 좋은 선택이었는가"를 그 상황의 맥락으로 설명하세요.
- 마지막은 반드시 "다음에 이런 상황이라면..." 스타일의 실천 조언으로 마무리하세요.
- 영어 표현을 섞지 마세요.
"""

BREAKOUT_CONTEXT = """당신은 Breakout 플레이어님에게 1:1 코칭을 해주는 게임 코치입니다.

역할:
- 플레이어님이 한 행동과 AI 에이전트가 선택한 행동을 비교하고, 에이전트의 선택이 왜 더 좋은 판단이었는지 설명합니다.
- 그 상황에서 어떻게 했어야 하는지 플레이어님이 다음 판에 바로 써먹을 수 있는 조언을 줍니다.
- 분석 보고서가 아니라, 옆에서 직접 말해주는 코치처럼 자연스럽게 설명하세요.

게임 정보:
- 플레이어님은 화면 아래 패들을 좌우로 움직여 공을 튕기고, 위쪽 벽돌을 모두 부수면 됩니다.
- 공을 놓치면 목숨을 잃습니다. 패들을 공 궤도에 미리 맞추는 것이 핵심입니다.
- 주요 행동: NOOP(대기), FIRE(공 발사), RIGHT(오른쪽), LEFT(왼쪽)

코칭 원칙:
- 플레이어님의 행동과 에이전트 행동을 반드시 대비해서 설명하세요.
- "에이전트가 왜 더 좋은 선택이었는가"를 공의 궤도, 득점 타이밍, 목숨 유지 관점에서 설명하세요.
- 마지막은 반드시 "다음에 이런 상황이라면..." 스타일의 실천 조언으로 마무리하세요.
- 영어 표현을 섞지 마세요.
"""

GOMOKU_CONTEXT = """당신은 오목 플레이어님에게 1:1 코칭을 해주는 게임 코치입니다.

역할:
- 플레이어님이 둔 수와 AI 에이전트가 권장한 수를 비교하고, 에이전트의 수가 왜 더 좋은 판단이었는지 설명합니다.
- 그 상황에서 어떤 수를 뒀어야 하는지 플레이어님이 다음 판에 바로 적용할 수 있는 조언을 줍니다.
- 분석 보고서가 아니라, 옆에서 직접 말해주는 코치처럼 자연스럽게 설명하세요.

게임 정보:
- 15×15 오목, 5목을 먼저 만들면 승리합니다.
- 핵심 개념: 열린 3목·4목, 양방향 확장, 핵심 자리 선점, 상대 4목 차단, 공격과 수비의 균형, 이후 수순 주도권.

코칭 원칙:
- 플레이어님이 둔 수와 에이전트 권장 수를 반드시 대비해서 설명하세요.
- "에이전트의 수가 왜 더 좋은가"를 그 자리의 전략적 가치(연결성, 확장성, 주도권)로 설명하세요.
- 마지막은 반드시 "다음에 이런 상황이라면..." 스타일의 실천 조언으로 마무리하세요.
- 좌표 숫자 나열보다 그 위치가 가진 의미를 설명하는 데 집중하세요.
- 영어 표현을 섞지 마세요.
"""


def sanitize_feedback_text(text: str) -> str:
    text = text.strip()
    # R1 등 reasoning 모델이 <think>...</think> 블록을 섞어 보낼 경우 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    text = re.sub(r"^(안녕하세요[!！.\s]*)", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?()\-\u3131-\u318E\uAC00-\uD7A3:\n]", "", text)
    replacements = {
        "거예요": "것입니다",
        "거에요": "것입니다",
        "했어요": "했습니다",
        "좋아요": "좋습니다",
        "보여요": "보였습니다",
        "살펴보자": "살펴보겠습니다",
        "다음과 같은 전략을 시도해 보면 유용할 수 있습니다.": "",
        "다음과 같은 전략을 시도해보면 유용할 수 있습니다.": "",
        "다음과 같은 전략을 시도해 보면 유용합니다.": "",
        "다음과 같은 전략을 시도해보면 유용합니다.": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_similar_sentence(a: str, b: str) -> bool:
    a_norm = re.sub(r"\s+", " ", a.strip())
    b_norm = re.sub(r"\s+", " ", b.strip())
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= 0.82


def format_feedback_text(text: str) -> str:
    text = sanitize_feedback_text(text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if sentences and not re.search(r"[.!?]$", sentences[-1]):
        sentences.pop()
    deduped = []
    for sentence in sentences:
        if any(is_similar_sentence(sentence, existing) for existing in deduped):
            continue
        deduped.append(sentence)
    sentences = deduped
    if len(sentences) <= 2:
        return "\n\n".join(sentences)
    # 두 문단으로 균등하게 분배 (4→2+2, 5→2+3, 6→3+3, 7→3+4 ...)
    mid = len(sentences) // 2
    return f"{' '.join(sentences[:mid])}\n\n{' '.join(sentences[mid:])}".strip()


def summarize_error(status_code: int | None, data: dict[str, Any] | None = None, exc: Exception | None = None) -> str:
    if status_code == 401:
        return "API 키가 올바르지 않거나 만료되었습니다."
    if status_code == 403:
        return "이 API 키는 현재 모델 호출 권한이 없습니다."
    if status_code == 404:
        return "선택한 무료 모델 엔드포인트를 찾지 못했습니다."
    if status_code == 429:
        return "무료 모델이 현재 혼잡하거나 rate limit 상태입니다."
    if status_code == 400:
        message = ""
        if data:
            message = str(data.get("error", {}).get("message", ""))
        if "reasoning" in message.lower():
            return "현재 선택한 모델은 일반 답변 대신 추론 전용 응답을 요구합니다."
        return "요청 형식이나 모델 설정이 현재 엔드포인트와 맞지 않습니다."
    if status_code and status_code >= 500:
        return "외부 LLM 제공자 쪽 서버 오류가 발생했습니다."
    if exc is not None:
        return f"네트워크 또는 응답 처리 오류가 발생했습니다: {exc}"
    return "외부 LLM이 현재 응답하지 않습니다."


def _format_game_state_section(gs: dict[str, Any]) -> str:
    """game_state에서 주목할 조건만 골라 한 줄 요약으로 반환합니다.
    정상 범위(적 다수·방패 온전·탄환 멀리)는 출력하지 않습니다."""
    if not gs:
        return ""

    flags: list[str] = []

    # 적 수 / 이동 속도 (빨라진 경우만)
    phase = gs.get('enemy_speed_phase', 'normal')
    ec = gs.get('enemy_count', 55)
    if phase == 'critical':
        flags.append(f"적 {ec}마리로 이동 속도 최고 단계")
    elif phase == 'fast':
        flags.append(f"적 {ec}마리로 이동 속도 빠름")

    # 게임오버 위협 (여유 35px 미만만)
    danger = gs.get('danger_distance', 999)
    if danger < 20:
        flags.append("적이 게임오버 직전까지 내려온 상태")
    elif danger < 35:
        flags.append(f"적이 {danger}px까지 근접")

    # 방패 (손상 또는 소실된 경우만)
    shields = gs.get('shield_integrity', [])
    meaningful = [p for p in shields if p >= 10]
    if shields and not meaningful:
        flags.append("방패 완전 소실")
    elif meaningful:
        avg = sum(meaningful) / len(meaningful)
        if avg < 40:
            flags.append("방패 심각 손상")
        elif avg < 70:
            flags.append("방패 일부 손상")
        # 70% 이상이면 정상 → 언급 생략

    # 적 탄환 근접 (8px 미만만)
    prox = gs.get('incoming_proximity', 23)
    if prox < 8:
        flags.append(f"적 탄환 {prox}px 근접")

    if not flags:
        return ""

    return f"\n주목할 상황: {' / '.join(flags)}."


def build_space_messages(summary: dict[str, Any], summary_lines: list[str]) -> list[dict[str, str]]:
    human_action_ko = ACTION_NAME_KO.get(summary["human_action_name"], summary["human_action_name"])
    agent_action_ko = ACTION_NAME_KO.get(summary["agent_action_name"], summary["agent_action_name"])
    gs_section = _format_game_state_section(summary.get('game_state', {}))

    # 스테이지 클리어 감지: 적이 극소수이고 첫 득점이 매우 빠른 경우
    gs = summary.get('game_state', {}) or {}
    ec = gs.get('enemy_count', 55)
    a_first = summary.get('agent_first_reward_step')
    if ec <= 3 and isinstance(a_first, (int, float)) and a_first <= 15:
        stage_note = (
            "\n※ 이 순간 남은 적이 극소수이고 에이전트 쪽 첫 득점이 매우 빠릅니다. "
            "이는 스테이지 마지막 적을 처치하며 스테이지가 클리어된 상황일 수 있습니다. "
            "득점을 '즉시 획득'처럼 표현하기보다, 스테이지 종료 시점의 위치 선택과 행동의 전략적 차이를 설명하세요."
        )
    else:
        stage_note = ""
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    if a_score > h_score:
        outcome_guidance = (
            f"이후 약 60프레임 비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
            f"이후 득점 프레임 데이터를 활용해 두 경로가 어떻게 달라졌는지 구체적으로 묘사하세요."
        )
    else:
        outcome_guidance = (
            f"이후 약 60프레임 비교 구간에서는 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
            f"대신 이 순간의 Q값 차이({summary.get('gap', 0):.4f})가 의미하는 전략적 판단 차이에 집중하세요. "
            f"필요하다면 '짧은 구간의 결과와 무관하게 이 순간의 선택 자체가 더 유리한 위치를 만드는 판단이었다'는 점을 자연스럽게 인정해도 좋습니다."
        )
    user_prompt = f"""다음은 Space Invaders 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {human_action_ko}
- 에이전트 행동: {agent_action_ko}
- 플레이어님 행동의 Q값: {summary.get('human_q', 0):.4f}
- 에이전트 행동의 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 이후 60프레임 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 60프레임 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 첫 득점 시점: {summary.get('human_first_reward_step', '득점 없음')}
- 에이전트 쪽 첫 득점 시점: {summary.get('agent_first_reward_step', '득점 없음')}
- 플레이어님 쪽 득점 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 득점 프레임들: {summary.get('agent_reward_steps') or '없음'}
{gs_section}
관찰 요약:
{chr(10).join(f"- {line}" for line in summary_lines)}

이후 경로 활용 지침: {outcome_guidance}{stage_note}

작성 방식:
- 반드시 정확히 두 문단으로 작성하세요. 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호(예: "첫째 문단", "분석", "1." 등)는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 첫 번째 문단 (3~4문장): 먼저 플레이어님이 한 행동을 언급하고 에이전트 행동과 명확히 대비하세요. 위의 "이후 경로 활용 지침"에 따라 이후 전개를 설명하세요. "주목할 상황"이 제공된 경우, 그 조건들을 억지로 나열하지 말고 상황을 이해하는 배경으로만 자연스럽게 활용하세요.
- 두 번째 문단 (2~3문장): "다음에 이런 상황이라면" 또는 그와 비슷한 뉘앙스로 시작해 구체적인 실천 조언을 주세요. 플레이어님이 바로 실천할 수 있게 안내하세요. 첫 번째 문단 내용을 반복하지 마세요.
- 점수와 득점 프레임은 전체 게임이 아닌 이 순간 이후 약 60프레임 비교 구간의 수치입니다. 점수를 언급할 때는 반드시 "이 구간에서" 또는 "비교 구간에서"라는 표현을 써서 전체 게임 점수로 오해하지 않도록 하세요.
- 한국어로만 쓰세요. 영어·일본어 등 외국어 표현은 쓰지 마세요.
- 말투는 "했습니다", "좋았습니다", "유리했습니다"처럼 단정한 존댓말로 쓰세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
"""
    return [
        {"role": "system", "content": SPACE_INVADERS_CONTEXT},
        {"role": "user", "content": user_prompt},
    ]


def build_breakout_messages(summary: dict[str, Any], summary_lines: list[str]) -> list[dict[str, str]]:
    human_action_ko = BO_ACTION_NAME_KO.get(summary["human_action_name"], summary["human_action_name"])
    agent_action_ko = BO_ACTION_NAME_KO.get(summary["agent_action_name"], summary["agent_action_name"])
    h_score = summary.get('human_score_delta', 0)
    a_score = summary.get('agent_score_delta', 0)
    if a_score > h_score:
        outcome_guidance = (
            f"이후 약 60프레임 비교 구간에서 에이전트({a_score:.0f}점)가 플레이어님({h_score:.0f}점)보다 높은 점수를 기록했습니다. "
            f"이후 득점 프레임 데이터를 활용해 두 경로가 어떻게 달라졌는지 구체적으로 묘사하세요."
        )
    else:
        outcome_guidance = (
            f"이후 약 60프레임 비교 구간에서는 플레이어님({h_score:.0f}점)의 점수가 에이전트({a_score:.0f}점)와 비슷하거나 더 높습니다. "
            f"이 구간 결과를 직접 비교 근거로 쓰지 마세요. "
            f"대신 이 순간의 Q값 차이({summary.get('gap', 0):.4f})가 의미하는 전략적 판단 차이에 집중하세요. "
            f"필요하다면 '짧은 구간의 결과와 무관하게 이 순간의 선택 자체가 더 유리한 위치를 만드는 판단이었다'는 점을 자연스럽게 인정해도 좋습니다."
        )
    user_prompt = f"""다음은 Breakout 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 플레이어님 행동: {human_action_ko}
- 에이전트 행동: {agent_action_ko}
- 플레이어님 행동의 Q값: {summary.get('human_q', 0):.4f}
- 에이전트 행동의 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 이후 60프레임 비교 구간 플레이어님 점수: {h_score:.1f}
- 이후 60프레임 비교 구간 에이전트 점수: {a_score:.1f}
- 플레이어님 쪽 첫 득점 시점: {summary.get('human_first_reward_step', '득점 없음')}
- 에이전트 쪽 첫 득점 시점: {summary.get('agent_first_reward_step', '득점 없음')}
- 플레이어님 쪽 득점 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 득점 프레임들: {summary.get('agent_reward_steps') or '없음'}

관찰 요약:
{chr(10).join(f"- {line}" for line in summary_lines)}

이후 경로 활용 지침: {outcome_guidance}

작성 방식:
- 반드시 정확히 두 문단으로 작성하세요. 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 첫 번째 문단 (3~4문장): 먼저 플레이어님이 한 행동을 언급하고 에이전트 행동과 명확히 대비하세요. 위의 "이후 경로 활용 지침"에 따라 이후 전개를 설명하세요. Q값 차이도 맥락으로 자연스럽게 녹이세요.
- 두 번째 문단 (2~3문장): "다음에 이런 상황이라면" 또는 그와 비슷한 뉘앙스로 시작해 구체적인 실천 조언을 주세요. 패들 위치, 공의 궤도 예측, 득점 타이밍 관점에서 플레이어님이 바로 실천할 수 있게 안내하세요. 첫 번째 문단 내용을 반복하지 마세요.
- 점수와 득점 프레임은 전체 게임이 아닌 이 순간 이후 약 60프레임 비교 구간의 수치입니다. 점수를 언급할 때는 반드시 "이 구간에서" 또는 "비교 구간에서"라는 표현을 써서 전체 게임 점수로 오해하지 않도록 하세요.
- 한국어로만 쓰세요. 영어 표현은 쓰지 마세요.
- 말투는 "했습니다", "좋았습니다", "유리했습니다"처럼 단정한 존댓말을 사용하세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
"""
    return [
        {"role": "system", "content": BREAKOUT_CONTEXT},
        {"role": "user", "content": user_prompt},
    ]


def build_gomoku_messages(summary: dict[str, Any], summary_lines: list[str]) -> list[dict[str, str]]:
    human_seq = ", ".join(summary.get("human_sequence_labels", [])[:8]) or "기록 없음"
    agent_seq = ", ".join(summary.get("agent_sequence_labels", [])[:8]) or "기록 없음"
    best_q = summary.get('best_q', 0)
    actual_q = summary.get('actual_q', 0)
    if best_q < 0.12:
        position_guidance = (
            f"권장 착수의 Q값도 {best_q:.4f}로 이 국면 자체가 이미 매우 불리한 상황입니다. "
            f"'에이전트가 이겼을 것이다'처럼 오해를 줄 수 있는 표현은 피하세요. "
            f"대신 그 어려운 상황에서 왜 에이전트의 수가 그나마 더 나은 선택이었는지 "
            f"(더 오래 버티기, 상대 실수 유도, 최소한의 확장성 유지)를 솔직하게 설명하세요. "
            f"상황의 어려움을 인정하면서도 그 안에서의 최선이 무엇인지 전달하세요."
        )
    else:
        position_guidance = (
            f"이후 수순 데이터를 활용해 두 경로가 실제로 어떻게 달라졌는지 묘사하세요. "
            f"인간 수순에서는 어떤 전개가 펼쳐졌고, 에이전트 수순에서는 어떤 이점이 생겼는지 구체적으로 보여주세요."
        )
    user_prompt = f"""다음은 오목 코칭 사례입니다. 아래 정보를 바탕으로 플레이어님에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 플레이어님 착수: ({summary['actual_row']}, {summary['actual_col']})
- 에이전트 권장 착수: ({summary['best_row']}, {summary['best_col']})
- 플레이어님 착수의 Q값: {actual_q:.4f}
- 권장 착수의 Q값: {best_q:.4f}
- 가치 차이: {summary.get('loss', 0):.4f}
- 플레이어님 쪽 이후 수순 (이 수를 둔 뒤 실제로 이어진 수들): {human_seq}
- 에이전트 쪽 이후 수순 (권장 수를 뒀을 때 시뮬레이션된 수들): {agent_seq}

관찰 요약:
{chr(10).join(f"- {line}" for line in summary_lines)}

이후 경로 활용 지침: {position_guidance}

작성 방식:
- 반드시 정확히 두 문단으로 작성하세요. 빈 줄 하나로 구분하세요.
- 문단 제목, 레이블, 번호는 절대 쓰지 마세요. 바로 본문으로 시작하세요.
- 첫 번째 문단 (3~4문장): 먼저 플레이어님이 둔 수를 언급하고 에이전트 권장 수와 명확히 대비하세요. 위의 "이후 경로 활용 지침"에 따라 이후 전개를 설명하세요. Q값 차이도 맥락으로 자연스럽게 녹이세요.
- 두 번째 문단 (2~3문장): "다음에 이런 상황이라면" 또는 그와 비슷한 뉘앙스로 시작해 구체적인 실천 조언을 주세요. 어떤 상황에서 어떤 자리를 우선 선택해야 하는지 플레이어님이 바로 실천할 수 있게 안내하세요. 첫 번째 문단 내용을 반복하지 마세요.
- 한국어로만 쓰세요. 영어 표현은 쓰지 마세요.
- 말투는 "했습니다", "좋았습니다", "유리했습니다"처럼 단정한 존댓말을 사용하세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
"""
    return [
        {"role": "system", "content": GOMOKU_CONTEXT},
        {"role": "user", "content": user_prompt},
    ]


def build_space_fallback_feedback(summary: dict[str, Any]) -> str:
    human_action_ko = ACTION_NAME_KO.get(summary["human_action_name"], summary["human_action_name"])
    agent_action_ko = ACTION_NAME_KO.get(summary["agent_action_name"], summary["agent_action_name"])
    human_steps = summary.get("human_reward_steps", [])
    agent_steps = summary.get("agent_reward_steps", [])
    human_steps_text = ", ".join(f"{s}프레임" for s in human_steps[:5]) if human_steps else "득점이 없었습니다"
    agent_steps_text = ", ".join(f"{s}프레임" for s in agent_steps[:5]) if agent_steps else "득점이 없었습니다"
    text = (
        f"이 상황에서는 인간 플레이어가 {human_action_ko}을 선택했지만, 에이전트의 {agent_action_ko}가 더 유리했습니다. "
        f"비교 영상을 보면 에이전트 쪽은 이동보다 공격 타이밍을 먼저 살리면서 득점 흐름을 앞당겼고, 인간 쪽은 위험을 피하는 데는 성공했지만 "
        f"득점으로 이어지는 기회를 더 늦게 잡았습니다.\n\n"
        f"인간 쪽 득점은 {human_steps_text}에 나왔고, 에이전트 쪽 득점은 {agent_steps_text}에 이어졌습니다. "
        f"즉 이번 장면에서 중요한 점은 최종 점수 차이만이 아니라, 에이전트가 더 이른 프레임부터 반복적으로 득점 기회를 만들었다는 점입니다."
    )
    return format_feedback_text(text)


def build_breakout_fallback_feedback(summary: dict[str, Any]) -> str:
    human_action_ko = BO_ACTION_NAME_KO.get(summary["human_action_name"], summary["human_action_name"])
    agent_action_ko = BO_ACTION_NAME_KO.get(summary["agent_action_name"], summary["agent_action_name"])
    human_steps = summary.get("human_reward_steps", [])
    agent_steps = summary.get("agent_reward_steps", [])
    human_steps_text = ", ".join(f"{s}프레임" for s in human_steps[:5]) if human_steps else "득점이 없었습니다"
    agent_steps_text = ", ".join(f"{s}프레임" for s in agent_steps[:5]) if agent_steps else "득점이 없었습니다"
    text = (
        f"이 상황에서는 인간 플레이어가 {human_action_ko}을 선택했지만, 에이전트의 {agent_action_ko}가 더 유리했습니다. "
        f"공의 궤도에 먼저 패들을 맞춰두는 에이전트의 움직임이 벽돌에 더 빠르게 공을 보내는 데 효과적이었습니다. "
        f"인간 쪽은 공을 놓치지 않는 데는 성공했지만, 득점 기회를 확보하는 타이밍이 늦었습니다.\n\n"
        f"인간 쪽 득점은 {human_steps_text}에 나왔고, 에이전트 쪽 득점은 {agent_steps_text}에 이어졌습니다. "
        f"이번 장면에서 중요한 점은 패들 이동의 방향과 타이밍이 공의 반사 각도와 벽돌 도달 속도를 크게 바꾼다는 것입니다."
    )
    return format_feedback_text(text)


def build_gomoku_fallback_feedback(summary: dict[str, Any]) -> str:
    actual = f"({summary['actual_row']}, {summary['actual_col']})"
    best = f"({summary['best_row']}, {summary['best_col']})"
    human_seq = ", ".join(summary.get("human_sequence_labels", [])[:8]) or "기록 없음"
    agent_seq = ", ".join(summary.get("agent_sequence_labels", [])[:8]) or "기록 없음"
    text = (
        f"이 장면에서는 인간 플레이어가 {actual}에 착수했지만, AI가 권장한 {best}가 더 유리했습니다. "
        f"실제 착수는 돌을 길게 이어 가거나 상대 위협을 먼저 끊어내는 힘이 다소 약했던 반면, 권장 착수는 더 높은 가치로 평가되어 이후 전개에서 주도권을 잡기 쉬운 선택이었습니다.\n\n"
        f"비교 보드를 보면 인간 쪽은 이후 수순이 {human_seq}로 이어졌고, AI 권장 수를 따른 쪽은 {agent_seq}처럼 더 빠르게 핵심 자리를 선점하면서 "
        f"열린 3목이나 4목으로 이어질 가능성을 넓혔습니다. 특히 권장 수는 한 방향만 잇는 데 그치지 않고 양쪽으로 확장될 여지를 만들거나, 상대가 먼저 위협을 만들기 전에 흐름을 끊는 데 더 적합한 수였습니다. "
        f"이번 장면에서 중요한 점은 한 수의 위치 차이가 돌의 연결, 상대 위협 차단, 다음 수를 강제하는 흐름, 이후 수순 전체의 효율을 함께 바꾼다는 점입니다."
    )
    return format_feedback_text(text)


def classify_model_route(selected_model: str, primary_model: str) -> str:
    if selected_model == primary_model:
        return "기본 모델"
    if selected_model == "openrouter/free":
        return "free 라우터"
    if selected_model in FALLBACK_MODELS:
        return "예비 모델"
    return "외부 모델"


def generate_feedback(game_type: str, summary: dict[str, Any]) -> tuple[str, str, str | None, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    if game_type == "space_invaders":
        gs = summary.get('game_state', {})
        summary_lines = [
            f"인간 행동은 {summary['human_action_name']}이고, 에이전트 행동은 {summary['agent_action_name']}입니다.",
            f"인간 쪽 득점 프레임은 {summary.get('human_reward_steps') or '없음'}입니다.",
            f"에이전트 쪽 득점 프레임은 {summary.get('agent_reward_steps') or '없음'}입니다.",
        ]
        # 게임 상태는 _format_game_state_section이 주목할 조건만 추려 gs_section으로 전달하므로
        # summary_lines에는 중복 포함하지 않음
        fallback_text = build_space_fallback_feedback(summary)
        messages = build_space_messages(summary, summary_lines)
    elif game_type == "breakout":
        summary_lines = [
            f"인간 행동은 {summary['human_action_name']}이고, 에이전트 행동은 {summary['agent_action_name']}입니다.",
            f"인간 쪽 득점 프레임은 {summary.get('human_reward_steps') or '없음'}입니다.",
            f"에이전트 쪽 득점 프레임은 {summary.get('agent_reward_steps') or '없음'}입니다.",
        ]
        fallback_text = build_breakout_fallback_feedback(summary)
        messages = build_breakout_messages(summary, summary_lines)
    else:  # gomoku
        summary_lines = [
            f"인간 착수는 ({summary['actual_row']}, {summary['actual_col']})입니다.",
            f"에이전트 권장 착수는 ({summary['best_row']}, {summary['best_col']})입니다.",
            f"한 수 차이가 이후 전개 전체의 효율에 영향을 줍니다.",
        ]
        fallback_text = build_gomoku_fallback_feedback(summary)
        messages = build_gomoku_messages(summary, summary_lines)

    if not api_key:
        return fallback_text, "local", None, "로컬"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "AI Arcade Strategic Advisor"),
    }
    site_url = os.getenv("OPENROUTER_SITE_URL")
    if site_url:
        headers["HTTP-Referer"] = site_url

    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    errors: list[str] = []
    for candidate_model in models_to_try:
        # reasoning 모델(DeepSeek R1 등)은 기본적으로 thinking 블록을 출력하므로
        # exclude:true 로 비활성화. 지원 안 하는 모델은 OpenRouter가 무시함.
        is_reasoning_model = any(x in candidate_model for x in ("r1", "r2", "qwq", "thinking"))
        body: dict[str, Any] = {
            "model": candidate_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1100,
        }
        if is_reasoning_model:
            body["reasoning"] = {"exclude": True}
        for attempt in range(MAX_LLM_ATTEMPTS):
            try:
                if requests is not None:
                    response = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=60)
                    status_code = response.status_code
                    ok = response.ok
                    data = response.json() if response.content else {}
                else:
                    req = urlrequest.Request(
                        OPENROUTER_URL,
                        data=json.dumps(body).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlrequest.urlopen(req, timeout=60) as resp:
                        status_code = getattr(resp, "status", 200)
                        ok = 200 <= status_code < 300
                        data = json.loads(resp.read().decode("utf-8"))

                if ok:
                    choice = data["choices"][0]["message"]
                    content = choice.get("content")
                    if isinstance(content, str) and content.strip():
                        actual_model = data.get("model", candidate_model)
                        return format_feedback_text(content), "llm", actual_model, classify_model_route(actual_model, model)
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                                parts.append(item["text"])
                        if parts:
                            actual_model = data.get("model", candidate_model)
                            return format_feedback_text("\n".join(parts)), "llm", actual_model, classify_model_route(actual_model, model)
                    finish_reason = data.get("choices", [{}])[0].get("finish_reason")
                    actual_model = data.get("model", candidate_model)
                    errors.append(
                        f"model={actual_model}, status=200, finish_reason={finish_reason}, 답변 본문이 없어 fallback 합니다."
                    )
                    break
                if status_code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                errors.append(f"model={candidate_model}, status={status_code}")
                break
            except urlerror.HTTPError as exc:
                if exc.code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                errors.append(f"model={candidate_model}, status={exc.code}")
                break
            except Exception as exc:
                if attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                errors.append(f"model={candidate_model}, error={exc}")
                break
    return fallback_text, "local", None, "로컬"


def test_openrouter_connection() -> tuple[bool, dict[str, Any]]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    if not api_key:
        return False, {"summary": "API 키가 설정되지 않았습니다.", "results": []}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "AI Arcade Strategic Advisor"),
    }
    site_url = os.getenv("OPENROUTER_SITE_URL")
    if site_url:
        headers["HTTP-Referer"] = site_url

    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    results: list[dict[str, str]] = []
    for candidate_model in models_to_try:
        is_reasoning_model = any(x in candidate_model for x in ("r1", "r2", "qwq", "thinking"))
        body: dict[str, Any] = {
            "model": candidate_model,
            "messages": [
                {"role": "system", "content": "당신은 한국어로 짧게 답하는 도우미입니다."},
                {"role": "user", "content": "연결 테스트입니다. '연결 성공'만 답하세요."},
            ],
            "temperature": 0,
            "max_tokens": 40,
        }
        if is_reasoning_model:
            body["reasoning"] = {"exclude": True}
        for attempt in range(MAX_LLM_ATTEMPTS):
            try:
                if requests is not None:
                    response = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=30)
                    status_code = response.status_code
                    ok = response.ok
                    data = response.json() if response.content else {}
                else:
                    req = urlrequest.Request(
                        OPENROUTER_URL,
                        data=json.dumps(body).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlrequest.urlopen(req, timeout=30) as resp:
                        status_code = getattr(resp, "status", 200)
                        ok = 200 <= status_code < 300
                        data = json.loads(resp.read().decode("utf-8"))

                if ok:
                    choice = data["choices"][0]["message"]
                    content = choice.get("content")
                    if isinstance(content, str) and content.strip():
                        return True, f"{candidate_model} 연결 성공"
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                                parts.append(item["text"])
                        if parts:
                            results.append({"model": candidate_model, "status": "success", "message": "연결 성공"})
                            return True, {"summary": f"{candidate_model} 연결 성공", "results": results}
                if status_code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                results.append({"model": candidate_model, "status": "error", "message": summarize_error(status_code, data)})
                break
            except urlerror.HTTPError as exc:
                if exc.code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                try:
                    data = json.loads(exc.read().decode("utf-8")) if exc.fp else {}
                except Exception:
                    data = {}
                results.append({"model": candidate_model, "status": "error", "message": summarize_error(exc.code, data)})
                break
            except Exception as exc:
                if attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                results.append({"model": candidate_model, "status": "error", "message": summarize_error(None, exc=exc)})
                break

    return False, {"summary": "모든 외부 LLM 후보 연결에 실패했습니다.", "results": results}