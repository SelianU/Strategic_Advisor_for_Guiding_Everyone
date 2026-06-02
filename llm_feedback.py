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
from difflib import SequenceMatcher
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


# ── 텍스트 정제 및 포맷 ──────────────────────────────────────────────────────

def sanitize_feedback_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^(안녕하세요[!！.\s]*)", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?()\-ㄱ-ㆎ가-힣:\n]", "", text)
    for old, new in {
        "거예요": "것입니다", "거에요": "것입니다",
        "했어요": "했습니다", "좋아요": "좋습니다",
        "보여요": "보였습니다", "살펴보자": "살펴보겠습니다",
        "다음과 같은 전략을 시도해 보면 유용할 수 있습니다.": "",
        "다음과 같은 전략을 시도해보면 유용합니다.": "",
    }.items():
        text = text.replace(old, new)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_similar_sentence(a: str, b: str) -> bool:
    a_norm = re.sub(r"\s+", " ", a.strip())
    b_norm = re.sub(r"\s+", " ", b.strip())
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= 0.82


def format_feedback_text(text: str) -> str:
    """LLM 응답을 2문단 서술형으로 정리합니다."""
    text = sanitize_feedback_text(text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if sentences and not re.search(r"[.!?]$", sentences[-1]):
        sentences.pop()
    deduped: list[str] = []
    for s in sentences:
        if any(_is_similar_sentence(s, e) for e in deduped):
            continue
        deduped.append(s)
    if len(deduped) <= 2:
        return "\n\n".join(deduped)
    mid = len(deduped) // 2
    return f"{' '.join(deduped[:mid])}\n\n{' '.join(deduped[mid:])}".strip()


def parse_structured_feedback(text: str) -> dict[str, str]:
    """하위 호환: full 텍스트를 structured dict 형태로 래핑합니다."""
    full = format_feedback_text(text)
    return {'situation': '', 'comparison': full, 'advice': '', 'full': full}


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


# ── 범용 빌더 (coach_config 기반) ────────────────────────────────────────────

def _build_atari_messages(game_id: str, summary: dict[str, Any]) -> list[dict[str, str]]:
    cfg = _load_coach_config(game_id)
    return [
        {"role": "system", "content": cfg.SYSTEM_PROMPT},
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
                        print(f'[LLM] ✅ 성공: {short_model_name(candidate)} ({len(content)}자)')
                        return content, actual_model, classify_model_route(actual_model, primary)
                    if isinstance(content, list):
                        parts = [i["text"] for i in content
                                 if isinstance(i, dict) and i.get("type") == "text" and i.get("text")]
                        if parts:
                            joined = "\n".join(parts)
                            print(f'[LLM] ✅ 성공(list): {short_model_name(candidate)} ({len(joined)}자)')
                            return joined, actual_model, classify_model_route(actual_model, primary)
                    print(f'[LLM] ⚠ 빈 응답: {short_model_name(candidate)} content={content!r}')
                    break  # 빈 응답 → 다음 모델

                print(f'[LLM] ✗ HTTP {status_code}: {short_model_name(candidate)} '
                      f'err={str((data or {}).get("error", {}).get("message", ""))[:80]}')
                if status_code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                break

            except urlerror.HTTPError as exc:
                print(f'[LLM] ✗ HTTPError {exc.code}: {short_model_name(candidate)}')
                if exc.code == 429 and attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                break
            except Exception as exc:
                print(f'[LLM] ✗ Exception({type(exc).__name__}): {short_model_name(candidate)} — {exc}')
                if attempt < MAX_LLM_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                break

    print('[LLM] ✗ 모든 모델 실패 → 로컬 폴백')
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
    print(f'[LLM] generate_feedback game_id={game_id} step={summary.get("step", "?")}')
    if _load_coach_config(game_id) is not None:
        messages      = _build_atari_messages(game_id, summary)
        fallback_text = _build_atari_fallback(game_id, summary)
    else:
        fallback_text = (
            "이 순간 플레이어님의 행동과 에이전트의 행동이 달랐습니다. "
            "에이전트의 Q값이 더 높아 더 유리한 판단이었습니다.\n\n"
            "다음에는 에이전트의 추천 행동을 참고해보세요."
        )
        fb = parse_structured_feedback(fallback_text)
        return fb['full'], fb, "local", None, "로컬 (coach_config 없음)"

    if not os.getenv("OPENROUTER_API_KEY"):
        print(f'[LLM] ⚠ OPENROUTER_API_KEY 미설정 → 로컬 폴백 (game_id={game_id})')
        fb = parse_structured_feedback(fallback_text)
        return fb['full'], fb, "local", None, "로컬"

    print(f'[LLM] API 키 확인됨, LLM 호출 시작 (game_id={game_id})')
    raw, model, route = _call_llm(messages)
    if raw:
        fb = parse_structured_feedback(raw)
        return fb['full'], fb, "llm", model, route

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
