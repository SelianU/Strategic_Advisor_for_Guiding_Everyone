# Breakout 체크포인트

학습된 D3QN 가중치 파일(`best_model.pth`)을 이 폴더에 배치하세요.

기대 경로: `ai_agents/breakout/checkpoints/best_model.pth`

저장 포맷은 Space Invaders와 동일합니다:

```python
torch.save({'q_network': net.state_dict(), 'frame': frame_count}, path)
```

로드 예시:

```python
from ai_agents.breakout import load_breakout_d3qn

net, n_actions = load_breakout_d3qn('ai_agents/breakout/checkpoints/best_model.pth', device='cuda')
```
