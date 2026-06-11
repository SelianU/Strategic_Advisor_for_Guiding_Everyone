"""
config_store.py — config.json (OpenRouter API 키 / 모델) 영속화.
"""
import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def is_valid_api_key(key: str) -> bool:
    """플레이스홀더('your own key' 등)나 빈 값을 실제 키로 오인하지 않도록 검사.
    실제 API 키에는 공백이 들어갈 수 없다."""
    key = (key or '').strip()
    return bool(key) and ' ' not in key


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            key = cfg.get('openrouter_api_key', '')
            if is_valid_api_key(key):
                os.environ['OPENROUTER_API_KEY'] = key.strip()
            elif key:
                print(f'⚠️  config.json 의 openrouter_api_key 가 유효한 키 형식이 아니어서 무시합니다: {key!r}')
            if cfg.get('openrouter_model'):
                os.environ['OPENROUTER_MODEL'] = cfg['openrouter_model']
        except Exception as e:
            print(f'⚠️  config.json 로드 실패: {e}')


def save_config(api_key: str, model: str):
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            pass
    cfg['openrouter_api_key'] = api_key
    cfg['openrouter_model']   = model
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)