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
    "arcee-ai/trinity-large-preview:free",
    "openrouter/free",
]
MAX_LLM_ATTEMPTS = 4
RETRY_DELAYS = [2.0, 4.0, 6.0]

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

SPACE_INVADERS_CONTEXT = """당신은 Space Invaders 플레이를 설명해주는 친절한 게임 코치입니다.

게임 정보:
- 플레이어는 화면 아래의 캐릭터를 좌우로 움직이며 적을 맞춥니다.
- 주요 행동은 NOOP, FIRE, LEFT, RIGHT, LEFTFIRE, RIGHTFIRE 입니다.
- 목표는 적을 맞혀 점수를 얻고, 위험한 탄이나 적의 공격을 피하면서 오래 살아남는 것입니다.
- 좋은 설명은 "무슨 버튼을 눌렀는가"보다 "그 선택이 이후 몇 프레임에서 어떤 결과를 만들었는가"를 자연스럽게 연결해줘야 합니다.

프로젝트 정보:
- 이 프로젝트는 강화학습 기반의 explainable AI 코칭 시스템입니다.
- D3QN 에이전트는 비교 기준이 되는 강한 정책입니다.
- 같은 게임 상태에서 인간의 행동과 에이전트가 더 선호하는 행동을 비교합니다.
- 선택된 장면은 가치 차이가 큰 장면으로, 에이전트 행동이 인간 행동보다 더 유리하다고 평가된 순간입니다.

설명 원칙:
- 연구 보고서처럼 쓰지 말고, 실제 플레이어에게 말해주듯 자연스럽고 따뜻한 한국어로 설명하세요.
- 점수 상승, 공격 기회, 위치 안정성, 생존 가능성 같은 눈에 보이는 변화를 중심으로 설명하세요.
- 영어 표현을 섞지 마세요.
"""

BREAKOUT_CONTEXT = """당신은 Breakout 플레이를 설명해주는 친절한 게임 코치입니다.

게임 정보:
- 플레이어는 화면 아래의 패들을 좌우로 움직여 공을 튕기고, 위쪽의 벽돌을 모두 부수는 것이 목표입니다.
- 주요 행동은 NOOP(대기), FIRE(공 발사), RIGHT(오른쪽), LEFT(왼쪽) 입니다.
- 공을 놓치면 목숨을 잃고, 벽돌을 빠르게 부술수록 높은 점수를 얻을 수 있습니다.
- 패들의 위치를 공의 궤도에 미리 맞추는 것이 핵심 기술입니다.
- 좋은 설명은 "무슨 버튼을 눌렀는가"보다 "그 선택이 이후 공의 궤도와 점수에 어떤 영향을 미쳤는가"를 자연스럽게 연결해줘야 합니다.

프로젝트 정보:
- 이 프로젝트는 강화학습 기반의 explainable AI 코칭 시스템입니다.
- D3QN 에이전트는 비교 기준이 되는 강한 정책입니다.
- 같은 게임 상태에서 인간의 행동과 에이전트가 더 선호하는 행동을 비교합니다.
- 선택된 장면은 가치 차이가 큰 장면으로, 에이전트 행동이 인간 행동보다 더 유리하다고 평가된 순간입니다.

설명 원칙:
- 연구 보고서처럼 쓰지 말고, 실제 플레이어에게 말해주듯 자연스럽고 따뜻한 한국어로 설명하세요.
- 패들 위치와 공의 궤도, 득점 타이밍, 목숨 유지 가능성 같은 눈에 보이는 변화를 중심으로 설명하세요.
- 영어 표현을 섞지 마세요.
"""

GOMOKU_CONTEXT = """당신은 오목 플레이를 설명해주는 친절한 게임 코치입니다.

게임 정보:
- 이 환경은 15×15 오목이며 5목을 먼저 만들면 승리합니다.
- 플레이어는 흑돌, 에이전트는 백돌 또는 반대로 같은 상태에서 더 좋은 한 수를 비교할 수 있습니다.
- 좋은 설명은 좌표 자체보다, 왜 그 자리가 더 좋은지와 이후 수순이 어떻게 달라지는지를 자연스럽게 알려주는 것입니다.

프로젝트 정보:
- 이 프로젝트는 강화학습 기반의 explainable AI 코칭 시스템입니다.
- PPO 정책이 비교 기준이 되는 강한 정책입니다.
- 같은 판 상태에서 인간 착수와 에이전트 권장 착수를 비교합니다.

설명 원칙:
- 연구 보고서처럼 쓰지 말고, 실제 플레이어에게 말해주듯 자연스럽고 따뜻한 한국어로 설명하세요.
- 돌의 연결 가능성, 열린 3목과 4목으로 이어질 가능성, 양방향 확장 가능성, 핵심 자리 선점, 상대 위협 차단, 공격과 수비의 균형 같은 내용을 중심으로 설명하세요.
- 가능하면 이 한 수가 공격을 만드는 수인지, 상대의 4를 막는 수인지, 다음 수를 강제하는 수인지까지 짚어주세요.
- 영어 표현을 섞지 마세요.
"""


def sanitize_feedback_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(안녕하세요[!！.\s]*)", "", text)
    text = re.sub(r"[^\w\s.,!?()\-\u3131-\u318E\uAC00-\uD7A3:\n]", "", text)
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
    if len(sentences) <= 4:
        return f"{' '.join(sentences[:2])}\n\n{' '.join(sentences[2:])}".strip()
    return (
        f"{' '.join(sentences[:2])}\n\n"
        f"{' '.join(sentences[2:4])}\n\n"
        f"{' '.join(sentences[4:])}"
    ).strip()


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


def build_space_messages(summary: dict[str, Any], summary_lines: list[str]) -> list[dict[str, str]]:
    human_action_ko = ACTION_NAME_KO.get(summary["human_action_name"], summary["human_action_name"])
    agent_action_ko = ACTION_NAME_KO.get(summary["agent_action_name"], summary["agent_action_name"])
    user_prompt = f"""다음은 Space Invaders 코칭 사례입니다. 아래 정보를 바탕으로 플레이어에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 인간 플레이어 행동: {human_action_ko}
- 에이전트 행동: {agent_action_ko}
- 인간 행동의 Q값: {summary.get('human_q', 0):.4f}
- 에이전트 행동의 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 인간 경로의 점수 변화: {summary.get('human_score_delta', 0):.1f}
- 에이전트 경로의 점수 변화: {summary.get('agent_score_delta', 0):.1f}
- 인간 쪽 첫 득점 시점: {summary.get('human_first_reward_step', '득점 없음')}
- 에이전트 쪽 첫 득점 시점: {summary.get('agent_first_reward_step', '득점 없음')}
- 인간 쪽 득점이 발생한 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 득점이 발생한 프레임들: {summary.get('agent_reward_steps') or '없음'}

관찰 요약:
{chr(10).join(f"- {line}" for line in summary_lines)}

작성 방식:
- 실제 플레이어에게 바로 조언해주는 말투로 써주세요.
- 인간 행동과 에이전트 행동을 꼭 비교해서 설명하세요.
- 점수가 더 빨리 올랐는지, 공격 기회를 더 잘 살렸는지, 더 안전했는지를 중심으로 설명하세요.
- 숫자를 그대로 반복하기보다, 그 숫자가 의미하는 바를 풀어서 설명하세요.
- "momentarily", "branch", "replay" 같은 영어 표현은 쓰지 마세요.
- 말투는 "했습니다", "좋았습니다", "유리했습니다"처럼 단정한 존댓말을 사용하세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
- 2~3문단 허용, 자세한 설명도 괜찮지만 문단을 나눠 가독성 있게 작성하세요.
"""
    return [
        {"role": "system", "content": SPACE_INVADERS_CONTEXT},
        {"role": "user", "content": user_prompt},
    ]


def build_breakout_messages(summary: dict[str, Any], summary_lines: list[str]) -> list[dict[str, str]]:
    human_action_ko = BO_ACTION_NAME_KO.get(summary["human_action_name"], summary["human_action_name"])
    agent_action_ko = BO_ACTION_NAME_KO.get(summary["agent_action_name"], summary["agent_action_name"])
    user_prompt = f"""다음은 Breakout 코칭 사례입니다. 아래 정보를 바탕으로 플레이어에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 스텝: {summary['step']}
- 인간 플레이어 행동: {human_action_ko}
- 에이전트 행동: {agent_action_ko}
- 인간 행동의 Q값: {summary.get('human_q', 0):.4f}
- 에이전트 행동의 Q값: {summary.get('agent_q', 0):.4f}
- 가치 차이: {summary.get('gap', 0):.4f}
- 인간 경로의 점수 변화: {summary.get('human_score_delta', 0):.1f}
- 에이전트 경로의 점수 변화: {summary.get('agent_score_delta', 0):.1f}
- 인간 쪽 첫 득점 시점: {summary.get('human_first_reward_step', '득점 없음')}
- 에이전트 쪽 첫 득점 시점: {summary.get('agent_first_reward_step', '득점 없음')}
- 인간 쪽 득점이 발생한 프레임들: {summary.get('human_reward_steps') or '없음'}
- 에이전트 쪽 득점이 발생한 프레임들: {summary.get('agent_reward_steps') or '없음'}

관찰 요약:
{chr(10).join(f"- {line}" for line in summary_lines)}

작성 방식:
- 실제 플레이어에게 바로 조언해주는 말투로 써주세요.
- 인간 행동과 에이전트 행동을 꼭 비교해서 설명하세요.
- 패들 위치, 공의 궤도 예측, 득점 타이밍, 목숨 유지 가능성을 중심으로 설명하세요.
- 숫자를 그대로 반복하기보다, 그 숫자가 의미하는 바를 풀어서 설명하세요.
- 영어 표현은 쓰지 마세요.
- 말투는 "했습니다", "좋았습니다", "유리했습니다"처럼 단정한 존댓말을 사용하세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
- 2~3문단 허용, 자세한 설명도 괜찮지만 문단을 나눠 가독성 있게 작성하세요.
"""
    return [
        {"role": "system", "content": BREAKOUT_CONTEXT},
        {"role": "user", "content": user_prompt},
    ]


def build_gomoku_messages(summary: dict[str, Any], summary_lines: list[str]) -> list[dict[str, str]]:
    human_seq = ", ".join(summary.get("human_sequence_labels", [])[:8]) or "기록 없음"
    agent_seq = ", ".join(summary.get("agent_sequence_labels", [])[:8]) or "기록 없음"
    user_prompt = f"""다음은 오목 코칭 사례입니다. 아래 정보를 바탕으로 플레이어에게 자연스러운 한국어 피드백을 작성해주세요.

상황 정보:
- 인간 플레이어 착수: ({summary['actual_row']}, {summary['actual_col']})
- 에이전트 권장 착수: ({summary['best_row']}, {summary['best_col']})
- 실제 착수의 Q값: {summary.get('actual_q', 0):.4f}
- 권장 착수의 Q값: {summary.get('best_q', 0):.4f}
- 가치 차이: {summary.get('loss', 0):.4f}
- 인간 쪽 이후 수순: {human_seq}
- 에이전트 쪽 이후 수순: {agent_seq}

관찰 요약:
{chr(10).join(f"- {line}" for line in summary_lines)}

작성 방식:
- 실제 플레이어에게 바로 조언해주는 말투로 써주세요.
- 인간 착수와 에이전트 권장 착수를 꼭 비교해서 설명하세요.
- 돌의 연결 가능성, 열린 3목과 4목으로 이어질 가능성, 양방향 확장 가능성, 핵심 자리 선점, 상대 위협 차단, 공격과 수비의 균형, 이후 수순에서의 주도권 변화를 중심으로 설명하세요.
- 가능하면 이 수가 공격을 만드는 수인지, 상대의 4를 막는 수인지, 다음 수를 강제하는 수인지, 중앙과 변 중 어느 쪽의 주도권을 잡는 수인지까지 설명하세요.
- 숫자를 그대로 반복하기보다, 그 숫자가 의미하는 바를 풀어서 설명하세요.
- 영어 표현은 쓰지 마세요.
- 말투는 "했습니다", "좋았습니다", "유리했습니다"처럼 단정한 존댓말을 사용하세요.
- 인삿말, 이모티콘, 과한 감탄사는 넣지 마세요.
- 2~3문단 허용, 자세한 설명도 괜찮지만 문단을 나눠 가독성 있게 작성하세요.
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
        summary_lines = [
            f"인간 행동은 {summary['human_action_name']}이고, 에이전트 행동은 {summary['agent_action_name']}입니다.",
            f"인간 쪽 득점 프레임은 {summary.get('human_reward_steps') or '없음'}입니다.",
            f"에이전트 쪽 득점 프레임은 {summary.get('agent_reward_steps') or '없음'}입니다.",
        ]
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
        body = {
            "model": candidate_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1100,
            "reasoning": {"enabled": False},
        }
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
        body = {
            "model": candidate_model,
            "messages": [
                {"role": "system", "content": "당신은 한국어로 짧게 답하는 도우미입니다."},
                {"role": "user", "content": "연결 테스트입니다. '연결 성공'만 답하세요."},
            ],
            "temperature": 0,
            "max_tokens": 40,
            "reasoning": {"enabled": False},
        }
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