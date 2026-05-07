"""
state.py (gomoku.state) — Gomoku 전역 상태 + PPO 모델 + 보드 상수.

리팩토링 핵심:
  기존에는 11개의 모듈 레벨 변수(gomoku_board, gomoku_history, …)를
  매 핸들러마다 `global ...` 선언으로 가변 사용했음.
  여기서는 단일 싱글턴 객체 `state` 의 속성으로 모아 둔다.

  사용 측에서는:
      from gomoku.state import state
      state.board   = ...
      state.active  = False
  처럼 쓰면 됨. `global` 선언이 더 이상 필요 없다.

  (싱글턴 객체의 속성 변경은 모든 임포트 사이트에서 즉시 반영된다.
   `from gomoku.state import gomoku_board` 식으로 변수만 가져오면 안 됨에 주의.)
"""
import os
from ai_agents.gomoku import load_gomoku_ppo
from extensions import DEVICE

# ── 보드 상수 ────────────────────────────────────────────────────────────────
BOARD_W, BOARD_H, N_IN_ROW = 15, 15, 5

# ── PPO 모델 로드 (모듈 임포트 시 1회) ───────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GOMOKU_PPO_PATH = os.path.join(
    _PROJECT_ROOT, 'ai_agents', 'gomoku', 'gomoku_rl',
    'pretrained_models', '15_15', 'ppo', '0.pt'
)
gomoku_net = load_gomoku_ppo(_GOMOKU_PPO_PATH, board_size=BOARD_W, device=DEVICE)
print('✅ Gomoku PPO 모델 로드 완료 (15×15)')


# ── 상태 싱글턴 ──────────────────────────────────────────────────────────────
class _GomokuState:
    """
    Gomoku 게임 진행 중 변하는 모든 상태를 한 곳에 모은다.

    기존 매핑:
      gomoku_board                → state.board
      gomoku_history              → state.history
      gomoku_active               → state.active
      ai_player                   → state.ai_player
      gomoku_analysis_results     → state.analysis_results
      gomoku_counterfactual_cache → state.counterfactual_cache
      gomoku_session_id           → state.session_id
      gomoku_analysis_timer       → state.analysis_timer
      gomoku_ai_turn_timer        → state.ai_turn_timer
      gomoku_state_seq            → state.state_seq
      gomoku_human_color          → state.human_color
    """
    def __init__(self):
        self.board                = None
        self.history              = []
        self.active               = False
        self.ai_player            = None
        self.analysis_results     = []
        self.counterfactual_cache = {}
        self.session_id           = 0
        self.analysis_timer       = None
        self.ai_turn_timer        = None
        self.state_seq            = 0
        self.human_color          = 1

    def cancel_timers(self):
        """진행 중인 분석/AI턴 타이머를 모두 안전하게 취소한다."""
        for timer in (self.analysis_timer, self.ai_turn_timer):
            if timer is not None:
                try:
                    timer.cancel()
                except Exception:
                    pass
        self.analysis_timer = self.ai_turn_timer = None


state = _GomokuState()