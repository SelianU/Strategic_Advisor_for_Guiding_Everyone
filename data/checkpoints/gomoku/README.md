# Gomoku 체크포인트 (참고)

Gomoku PPO 모델은 git 서브모듈 `ai_agents/gomoku/gomoku_rl/` 안에 있습니다:

```
ai_agents/gomoku/gomoku_rl/pretrained_models/15_15/ppo/0.pt
```

서브모듈을 처음 받을 때:

```bash
git submodule update --init --recursive
```

다른 보드 사이즈/모델을 사용하려면 `app.py`의 `_GOMOKU_PPO_PATH` 상수를 수정하세요.
