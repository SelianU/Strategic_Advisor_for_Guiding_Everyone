"""
llm_feedback.py
───────────────
게임별 코칭 피드백 생성기.

각 게임의 ai_agents/{game_id}/coach_config.py에서 설정을 동적으로 읽어
LLM 또는 로컬 폴백 피드백을 생성합니다.

coach_config.py 필수 항목:
    SYSTEM_PROMPT              : str
    action_name_ko(name)       : callable → str
    build_outcome_guidance(summary) : callable → str
    build_user_prompt(summary)      : callable → str
    build_fallback_feedback(summary): callable → str

선택 항목:
    reward_label : str  (기본 "득점")
"""

import importlib
import json
import os
import re
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    import requests  # type: ignore
except Exception:
    requests = None


# ── OpenRouter 설정 ───────────────────────────────────────────────────────────

OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL   = "meta-llama/llama-3.3-70b-instruct:free"
FALLBACK_MODELS = [
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-3-27b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-12b-it:free",
]
FALLBACK_PRIORITY = FALLBACK_MODELS[:2]
FALLBACK_POOL     = FALLBACK_MODELS[2:]

MAX_LLM_ATTEMPTS = 2
RETRY_DELAYS     = [2.0, 4.0]


def short_model_name(full: str) -> str:
    name = full.split('/')[-1].replace(':free', '')
    for suffix in ('-instruct', '-it'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


# ── 공통 출력 형식 규칙 ───────────────────────────────────────────────────────

_STRUCTURED_OUTPUT_RULE = """
출력 형식 (반드시 준수):
답변은 아래 세 섹션으로 구성하고, 각 섹션을 정확히 해당 태그로 시작하세요.
태그는 줄의 맨 앞에 단독으로 위치해야 합니다.

[상황]
이 순간 어떤 상황이었는지 1~2문장으로 서술합니다.

[비교]
플레이어님의 행동/착수와 에이전트의 선택을 대비하고 왜 에이전트가 더 유리했는지 2~4문장으로 설명합니다.

[조언]
"다음에 이런 상황이라면" 또는 유사한 뉘앙스로 시작하는 실천 조언을 1~2문장으로 씁니다.

규칙:
- 태그 외에 다른 레이블, 제목, 번호는 절대 쓰지 마세요.
- 한국어로만 쓰세요. 영어 표현 금지.
- 말투는 단정한 존댓말 ("했습니다", "좋았습니다").
- 인삿말, 이모티콘, 감탄사 금지.
"""

# ── Gomoku 시스템 프롬프트 (board game이라 별도 유지) ─────────────────────────

GOMOKU_SYSTEM_PROMPT = (
    "당신은 오목 플레이어님에게 1:1 코칭을 해주는 게임 코치입니다.\n\n"
    "게임 정보:\n"
    "- 15×15 오목, 5목을 먼저 만들면 승리합니다.\n"
    "- 핵심 개념: 열린 3목·4목, 양방향 확장, 핵심 자리 선점, 상대 4목 차단, 공격과 수비의 균형.\n"
    + _STRUCTURED_OUTPUT_RULE
)


# ── coach_config 동적 임포트 ─────────────────────────────────────────────────

_config_cache: dict[str, Any] = {}


def _load_coach_config(game_id: str) -> Any | None:
    """ai_agents/{game_id}/coach_config.py를 동적으로 임포트해 캐싱합니다."""
    if game_id in _config_cache:
        return _config_cache[game_id]
    try:
        cfg = importlib.import_module(f"ai_agents.{game_id}.coach_config")
        _config_cache[game_id] = cfg
        return cfg
    except ModuleNotFoundError:
        _config_cache[game_id] = None
        return None


# ── 구조화 피드백 파싱 ────────────────────────────────────────────────────────

def sanitize_feedback_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^(안녕하세요[!！.\s]*)", "", text)
    # 태그 문자 []를 허용하도록 정규식 수정
    text = re.sub(r"[^a-zA-Z0-9\s.,!?()\-\u3131-\u318E\uAC00-\uD7A3:\n\[\]]", "", text)
    for old, new in {
        "거예요": "것입니다", "거에요": "것입니다",
        "했어요": "했습니다", "좋아요": "좋습니다",
        "보여요": "보였습니다", "살펴보자": "살펴보겠습니다",
    }.items():
        text = text.replace(old, new)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_structured_feedback(text: str) -> dict[str, str]:
    """
    [상황] / [비교] / [조언] 태그를 파싱합니다.
    태그가 없으면 전체를 comparison으로 폴백합니다.

    Returns: {'situation', 'comparison', 'advice', 'full'}
    """
    tags = {'situation': '[상황]', 'comparison': '[비교]', 'advice': '[조언]'}
    positions: dict[str, int] = {
        key: text.find(tag)
        for key, tag in tags.items()
        if text.find(tag) != -1
    }

    if not positions:
        clean = sanitize_feedback_text(text)
        return {'situation': '', 'comparison': clean, 'advice': '', 'full': clean}

    order = sorted(positions.items(), key=lambda x: x[1])
    sections: dict[str, str] = {}
    for i, (key, pos) in enumerate(order):
        start = pos + len(tags[key])
        end   = order[i + 1][1] if i + 1 < len(order) else len(text)
        sections[key] = sanitize_feedback_text(text[start:end].strip())

    situation  = sections.get('situation', '')
    comparison = sections.get('comparison', '')
    advice     = sections.get('advice', '')
    full       = '\n\n'.join(p for p in [situation, comparison, advice] if p)
    return {'situation': situation, 'comparison': comparison, 'advice': advice, 'full': full}


def format_feedback_text(text: str) -> str:
    """하위 호환: full 텍스트만 반환."""
    return parse_structured_feedback(text)['full']


# ── 오류 요약 ─────────────────────────────────────────────────────────────────

def summarize_error(status_code: int | None,
                    data: dict[str, Any] | None = None,
                    exc: Exception | None = None) -> str:
    if status_code == 401: return "API 키가 올바르지 않거나 만료되었습니다."
    if status_code == 403: return "이 API 키는 현재 모델 호출 권한이 없습니다."
    if status_code == 404: return "선택한 무료 모델 엔드포인트를 찾지 못했습니다."
    if status_code == 429: return "무료 모델이 현재 혼잡하거나 rate limit 상태입니다."
    if status_code == 400:
        msg = str((data or {}).get("error", {}).get("message", ""))
        return ("현재 선택한 모델은 추론 전용 응답을 요구합니다."
                if "reasoning" in msg.lower()
                else "요청 형식이나 모델 설정이 현재 엔드포인트와 맞지 않습니다.")
    if status_code and status_code >= 500: return "외부 LLM 제공자 쪽 서버 오류가 발생했습니다."
    if exc: return f"네트워크 또는 응답 처리 오류가 발생했습니다: {exc}"
    return "외부 LLM이 현재 응답하지 않습니다."


def classify_model_route(selected: str, primary: str) -> str:
    if selected == primary:           return "기본 모델"
    if selected == "openrouter/free": return "free 라우터"
    if selected in FALLBACK_MODELS:   return "예비 모델"
    return "외부 모델"


# ── Gomoku 전용 빌더 ─────────────────────────────────────────────────────────

def _build_gomoku_messages(summary: dict[str, Any]) -> list[dict[str, str]]:
    best_q    = summary.get('best_q', 0)
    actual_q  = summary.get('actual_q', 0)
    human_seq = ", ".join(summary.get("human_sequence_labels", [])[:8]) or "기록 없음"
    agent_seq = ", ".join(summary.get("agent_sequence_labels", [])[:8]) or "기록 없음"
    path_guide = (
        f"권장 착수의 Q값도 {best_q:.4f}로 이 국면 자체가 매우 불리합니다. "
        "에이전트가 이겼을 것이라는 표현은 피하고, 어려운 상황에서 왜 에이전트의 수가 그나마 더 나은지 설명하세요."
        if best_q < 0.12 else
        f"이후 수순 데이터를 활용해 두 경로가 어떻게 달라졌는지 묘사하세요. "
        f"인간 수순({human_seq})과 에이전트 수순({agent_seq})의 차이를 구체적으로 보여주세요."
    )
    user_prompt = (
        f"다음은 오목 코칭 사례입니다.\n\n"
        f"상황 정보: ※ 좌표는 (위에서 몇 번째 행, 왼쪽에서 몇 번째 열) — 1-인덱스 기준\n"
        f"- 플레이어님 착수: ({summary['actual_row']}, {summary['actual_col']})\n"
        f"- 에이전트 권장 착수: ({summary['best_row']}, {summary['best_col']})\n"
        f"- 플레이어님 Q값: {actual_q:.4f} / 에이전트 Q값: {best_q:.4f}\n"
        f"- 가치 차이: {summary.get('loss', 0):.4f}\n"
        f"- 플레이어님 이후 수순: {human_seq}\n"
        f"- 에이전트 이후 수순: {agent_seq}\n\n"
        f"이후 경로 지침: {path_guide}\n"
    )
    return [
        {"role": "system", "content": GOMOKU_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]


def _build_gomoku_fallback(summary: dict[str, Any]) -> str:
    actual    = f"({summary['actual_row']}, {summary['actual_col']})"
    best      = f"({summary['best_row']}, {summary['best_col']})"
    human_seq = ", ".join(summary.get("human_sequence_labels", [])[:8]) or "기록 없음"
    agent_seq = ", ".join(summary.get("agent_sequence_labels", [])[:8]) or "기록 없음"
    return (
        f"[상황]\n플레이어님은 {actual}에 착수했지만, AI가 권장한 {best}가 더 유리했습니다.\n\n"
        f"[비교]\n실제 착수는 돌을 길게 잇거나 상대 위협을 끊어내는 힘이 다소 약했던 반면, "
        f"권장 착수는 이후 주도권을 잡기 쉬운 선택이었습니다. "
        f"인간 수순이 {human_seq}로 이어진 데 비해, 에이전트 수순 {agent_seq}는 "
        f"더 빠르게 핵심 자리를 선점했습니다.\n\n"
        f"[조언]\n다음에 이런 상황이라면, 한 방향만 잇는 수보다 양쪽으로 확장될 여지가 있는 자리를 먼저 찾아보세요."
    )


# ── Atari 범용 빌더 (coach_config 기반) ──────────────────────────────────────

def _build_atari_messages(game_id: str, summary: dict[str, Any]) -> list[dict[str, str]]:
    cfg = _load_coach_config(game_id)
    return [
        {"role": "system", "content": cfg.SYSTEM_PROMPT + _STRUCTURED_OUTPUT_RULE},
        {"role": "user",   "content": cfg.build_user_prompt(summary)},
    ]


def _build_atari_fallback(game_id: str, summary: dict[str, Any]) -> str:
    return _load_coach_config(game_id).build_fallback_feedback(summary)


# ── LLM 호출 공통 로직 ───────────────────────────────────────────────────────

def _call_llm(messages: list[dict[str, str]]) -> tuple[str | None, str | None, str]:
    """
    LLM을 호출합니다.

    Returns:
        (content, model_name, route) — 실패 시 (None, None, '로컬')
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None, None, "로컬"

    primary = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "AI Arcade Strategic Advisor"),
    }
    site_url = os.getenv("OPENROUTER_SITE_URL")
    if site_url:
        headers["HTTP-Referer"] = site_url

    for candidate in [primary] + [m for m in FALLBACK_MODELS if m != primary]:
        is_reasoning = any(x in candidate for x in ("r1", "r2", "qwq", "thinking"))
        body: dict[str, Any] = {
            "model": candidate, "messages": messages,
            "temperature": 0.2, "max_tokens": 1100,
        }
        if is_reasoning:
            body["reasoning"] = {"exclude": True}

        for attempt in range(MAX_LLM_ATTEMPTS):
            try:
                if requests is not None:
                    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=60)
                    status_code, ok = resp.status_code, resp.ok
                    data = resp.json() if resp.content else {}
                else:
                    req = urlrequest.Request(
                        OPENROUTER_URL, data=json.dumps(body).encode(),
                        headers=headers, method="POST",
                    )
                    with urlrequest.urlopen(req, timeout=60) as r:
                        status_code = getattr(r, "status", 200)
                        ok = 200 <= status_code < 300
                        data = json.loads(r.read().decode())

                if ok:
                    actual_model = data.get("model", candidate)
                    content = data["choices"][0]["message"].get("content")
                    if isinstance(content, str) and content.strip():
                        return content, actual_model, classify_model_route(actual_model, primary)
                    if isinstance(content, list):
                        parts = [i["text"] for i in content
                                 if isinstance(i, dict) and i.get("type") == "text" and i.get("text")]
                        if parts:
                            return "\n".join(parts), actual_model, classify_model_route(actual_model, primary)
                    break  # 빈 응답 → 다음 모델

                if status_code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                break

            except urlerror.HTTPError as exc:
                if exc.code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                break
            except Exception:
                if attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                break

    return None, None, "로컬"


# ── 공개 API ─────────────────────────────────────────────────────────────────

def generate_feedback(
    game_id: str,
    summary: dict[str, Any],
) -> tuple[str, dict[str, str], str, str | None, str]:
    """
    game_id에 맞는 코칭 피드백을 생성합니다.

    Returns:
        (full_text, structured_dict, source, model, route)
    """
    # ── 메시지 / 폴백 결정 ──────────────────────────────────────────────────
    if game_id == "gomoku":
        messages      = _build_gomoku_messages(summary)
        fallback_text = _build_gomoku_fallback(summary)
    elif _load_coach_config(game_id) is not None:
        messages      = _build_atari_messages(game_id, summary)
        fallback_text = _build_atari_fallback(game_id, summary)
    else:
        # coach_config 없음 → 최소 폴백
        fallback_text = (
            "[상황]\n이 순간 플레이어님의 행동과 에이전트의 행동이 달랐습니다.\n\n"
            "[비교]\n에이전트의 Q값이 더 높아 더 유리한 판단이었습니다.\n\n"
            "[조언]\n다음에는 에이전트의 추천 행동을 참고해보세요."
        )
        fb = parse_structured_feedback(fallback_text)
        return fb['full'], fb, "local", None, "로컬 (coach_config 없음)"

    # ── API 키 없음 → 로컬 폴백 ────────────────────────────────────────────
    if not os.getenv("OPENROUTER_API_KEY"):
        fb = parse_structured_feedback(fallback_text)
        return fb['full'], fb, "local", None, "로컬"

    # ── LLM 호출 ────────────────────────────────────────────────────────────
    raw, model, route = _call_llm(messages)
    if raw:
        fb = parse_structured_feedback(raw)
        return fb['full'], fb, "llm", model, route

    # ── 모든 모델 실패 → 로컬 폴백 ─────────────────────────────────────────
    fb = parse_structured_feedback(fallback_text)
    return fb['full'], fb, "local", None, "로컬"


def test_openrouter_connection() -> tuple[bool, dict[str, Any]]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    primary = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    if not api_key:
        return False, {"summary": "API 키가 설정되지 않았습니다.", "results": []}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "AI Arcade Strategic Advisor"),
    }
    site_url = os.getenv("OPENROUTER_SITE_URL")
    if site_url:
        headers["HTTP-Referer"] = site_url

    results: list[dict[str, str]] = []
    for candidate in [primary] + [m for m in FALLBACK_MODELS if m != primary]:
        is_reasoning = any(x in candidate for x in ("r1", "r2", "qwq", "thinking"))
        body: dict[str, Any] = {
            "model": candidate,
            "messages": [
                {"role": "system", "content": "당신은 한국어로 짧게 답하는 도우미입니다."},
                {"role": "user",   "content": "연결 테스트입니다. '연결 성공'만 답하세요."},
            ],
            "temperature": 0, "max_tokens": 40,
        }
        if is_reasoning:
            body["reasoning"] = {"exclude": True}

        for attempt in range(MAX_LLM_ATTEMPTS):
            try:
                if requests is not None:
                    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=30)
                    status_code, ok = resp.status_code, resp.ok
                    data = resp.json() if resp.content else {}
                else:
                    req = urlrequest.Request(
                        OPENROUTER_URL, data=json.dumps(body).encode(),
                        headers=headers, method="POST",
                    )
                    with urlrequest.urlopen(req, timeout=30) as r:
                        status_code = getattr(r, "status", 200)
                        ok = 200 <= status_code < 300
                        data = json.loads(r.read().decode())

                if ok:
                    content = data["choices"][0]["message"].get("content")
                    if isinstance(content, str) and content.strip():
                        return True, {"summary": f"{candidate} 연결 성공", "results": results}
                    if isinstance(content, list):
                        parts = [i["text"] for i in content
                                 if isinstance(i, dict) and i.get("type") == "text" and i.get("text")]
                        if parts:
                            return True, {"summary": f"{candidate} 연결 성공", "results": results}

                if status_code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                results.append({"model": candidate, "status": "error",
                                "message": summarize_error(status_code, data)})
                break

            except urlerror.HTTPError as exc:
                if exc.code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                try:
                    err_data = json.loads(exc.read().decode()) if exc.fp else {}
                except Exception:
                    err_data = {}
                results.append({"model": candidate, "status": "error",
                                "message": summarize_error(exc.code, err_data)})
                break
            except Exception as exc:
                if attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                results.append({"model": candidate, "status": "error",
                                "message": summarize_error(None, exc=exc)})
                break

    return False, {"summary": "모든 외부 LLM 후보 연결에 실패했습니다.", "results": results}