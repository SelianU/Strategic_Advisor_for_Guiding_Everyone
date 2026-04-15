"""
gomoku_rl_adapter.py
AlphaZero_Gomoku Board / MCTSPlayer 인터페이스를 gomoku_rl(DQN/PPO) 위에서 제공하는 어댑터.
app.py의 분석 코드(compute_q_values, run_gomoku_analysis, counterfactual 등)를 변경 없이 재사용한다.
"""

import sys
import os
import copy
import numpy as np
import torch
from tensordict import TensorDict
try:
    # torchrl >= 0.7
    from torchrl.data import Categorical as _ActionSpec
    from torchrl.data import Composite as _CompositeSpec
    from torchrl.data import Unbounded as _ObsSpec
    from torchrl.data import Binary as _MaskSpec
except ImportError:
    # torchrl < 0.7 (deprecated names)
    from torchrl.data.tensor_specs import DiscreteTensorSpec as _ActionSpec  # noqa
    from torchrl.data.tensor_specs import CompositeSpec as _CompositeSpec    # noqa
    from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec as _ObsSpec  # noqa
    from torchrl.data.tensor_specs import BinaryDiscreteTensorSpec as _MaskSpec  # noqa
from omegaconf import OmegaConf

GOMOKU_RL_DIR = os.path.join(os.path.dirname(__file__), "gomoku_rl")
if GOMOKU_RL_DIR not in sys.path:
    sys.path.insert(0, GOMOKU_RL_DIR)

from gomoku_rl.core import Gomoku
from gomoku_rl.policy import get_policy


# ─── 스펙 헬퍼 ──────────────────────────────────────────────────────────────

def _make_specs(board_size: int, device):
    n_actions = board_size * board_size
    action_spec = _ActionSpec(n_actions, shape=[1], device=device)
    observation_spec = _CompositeSpec(
        {
            "observation": _ObsSpec(
                device=device, shape=[1, 3, board_size, board_size]
            ),
            "action_mask": _MaskSpec(
                n=n_actions,
                device=device,
                shape=[1, n_actions],
                dtype=torch.bool,
            ),
        },
        shape=[1],
        device=device,
    )
    return action_spec, observation_spec


# ─── 모델 로더 ───────────────────────────────────────────────────────────────

def load_gomoku_dqn(checkpoint_path: str, board_size: int = 15, device="cpu"):
    """
    DQN pretrained 모델을 로드해서 GomokuDQNEngine을 반환한다.
    학습에 필요한 replay buffer는 생성하지 않고 actor만 로드한다.
    """
    from gomoku_rl.policy.common import make_dqn_actor
    dqn_cfg_path = os.path.join(GOMOKU_RL_DIR, "cfg", "baseline", "dqn.yaml")
    cfg = OmegaConf.load(dqn_cfg_path)
    action_spec, _ = _make_specs(board_size, device)
    actor = make_dqn_actor(cfg, action_spec, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(state_dict["actor"])
    actor.eval()
    return GomokuDQNEngine(actor, board_size=board_size, device=device)


# ─── 보드 클래스 ──────────────────────────────────────────────────────────────

class GomokuRLBoard:
    """
    AlphaZero Board 호환 인터페이스.

    공개 속성/메서드:
        width, height, n_in_row
        states      : {move_int: player}  — 1=흑, 2=백
        availables  : 착수 가능한 위치 목록
        current_player : 1 or 2
        last_move   : 마지막 착수 위치 (int, 없으면 -1)
        init_board(start_player)
        do_move(move)
        game_end()  → (done: bool, winner: int)  — winner: 1/2 또는 -1(무승부)
    """

    def __init__(self, board_size: int = 15, n_in_row: int = 5, device="cpu"):
        self.width = board_size
        self.height = board_size
        self.n_in_row = n_in_row
        self._device = device
        self._env = Gomoku(num_envs=1, board_size=board_size, device=device)
        self._env.reset()
        self.states: dict = {}  # {move_int: player (1 or 2)}

    def init_board(self, start_player: int = 0):
        self._env.reset()
        self.states = {}

    # ── 읽기 전용 프로퍼티 ───────────────────────────────────────────────────

    @property
    def current_player(self) -> int:
        turn = self._env.turn[0].item()
        return 1 if turn == 0 else 2  # turn 0 = 흑 차례

    @property
    def availables(self) -> list:
        mask = self._env.get_action_mask()[0]  # (board_size^2,) bool
        return [i for i, v in enumerate(mask.tolist()) if v]

    @property
    def last_move(self) -> int:
        return int(self._env.last_move[0].item())

    # ── 착수 ────────────────────────────────────────────────────────────────

    def do_move(self, move: int):
        player = self.current_player
        self._env.step(torch.tensor([move], device=self._device))
        self.states[move] = player

    # ── 종료 판정 ──────────────────────────────────────────────────────────

    def game_end(self):
        done = self._env.done[0].item()
        if not done:
            return False, -1
        # 무승부: 보드가 가득 참
        if self._env.move_count[0].item() >= self.width * self.height:
            return True, -1
        # 마지막으로 착수한 플레이어가 승자
        # do_move 후 turn이 이미 다음 플레이어로 넘어간 상태
        current_turn = self._env.turn[0].item()  # 0=흑 차례
        winner = 1 if current_turn == 1 else 2   # 현재 흰 차례면 → 흑이 방금 이김
        return True, winner

    # ── deepcopy 지원 ──────────────────────────────────────────────────────

    def __deepcopy__(self, memo):
        new_obj = GomokuRLBoard.__new__(GomokuRLBoard)
        new_obj.width = self.width
        new_obj.height = self.height
        new_obj.n_in_row = self.n_in_row
        new_obj._device = self._device
        # Gomoku 내부 상태를 텐서 단위로 복사
        new_obj._env = Gomoku(num_envs=1, board_size=self.width, device=self._device)
        new_obj._env.board = self._env.board.clone()
        new_obj._env.done = self._env.done.clone()
        new_obj._env.turn = self._env.turn.clone()
        new_obj._env.move_count = self._env.move_count.clone()
        new_obj._env.last_move = self._env.last_move.clone()
        new_obj.states = dict(self.states)
        return new_obj


# ─── DQN 엔진 ─────────────────────────────────────────────────────────────────

class GomokuDQNEngine:
    """
    AlphaZero PolicyValueNetNumpy + MCTSPlayer 역할을 DQN actor로 대체.
    replay buffer 등 학습 관련 객체는 일절 생성하지 않는다.

    공개 메서드:
        policy_value_fn(board) → (act_probs_iter, v_s)
            act_probs_iter : zip(positions, q_values)  — compute_q_values 호환
            v_s            : 합법 착수 중 최대 Q-value
        get_action(board, temp=1e-3) → int
            MCTSPlayer.get_action 호환
    """

    def __init__(self, actor, board_size: int = 15, device="cpu"):
        self._actor = actor   # QValueActor (학습용 DQN 전체가 아닌 actor만)
        self._board_size = board_size
        self._device = device

    def _make_td(self, board: GomokuRLBoard) -> TensorDict:
        obs = board._env.get_encoded_board().to(self._device)   # (1, 3, B, B)
        mask = board._env.get_action_mask().to(self._device)    # (1, B*B) bool
        return TensorDict(
            {"observation": obs, "action_mask": mask},
            batch_size=[1],
            device=self._device,
        )

    def _raw_q(self, board: GomokuRLBoard) -> np.ndarray:
        """합법 위치에만 Q-value를 채운 길이 board_size^2 배열을 반환. 불법 위치는 NaN."""
        with torch.no_grad():
            result = self._actor(self._make_td(board))
        q = result["action_value"][0].cpu().float().numpy()     # (B*B,)
        q_out = np.full(self._board_size ** 2, np.nan, dtype=np.float32)
        for pos in board.availables:
            q_out[pos] = float(q[pos])
        return q_out

    def policy_value_fn(self, board: GomokuRLBoard):
        """compute_q_values()가 기대하는 (iter[(pos, q)], v_s) 형태로 반환."""
        legal = board.availables
        if not legal:
            return iter([]), 0.0
        q = self._raw_q(board)
        pairs = [(pos, float(q[pos])) for pos in legal]
        v_s = float(max(q[pos] for pos in legal))
        return iter(pairs), v_s

    def get_action(self, board: GomokuRLBoard, temp: float = 1e-3) -> int:
        """가장 높은 Q-value의 합법 착수를 반환. MCTSPlayer.get_action 호환."""
        with torch.no_grad():
            result = self._actor(self._make_td(board))
        return int(result["action"][0].item())


# ─── PPO 엔진 ─────────────────────────────────────────────────────────────────

def load_gomoku_ppo(checkpoint_path: str, board_size: int = 15, device="cpu"):
    """
    PPO pretrained 모델을 로드해서 GomokuPPOEngine을 반환한다.
    share_network=True 구조: actor가 backbone(hidden)을 계산하고 critic이 이를 받는다.
    """
    from gomoku_rl.policy.common import make_ppo_ac
    ppo_cfg_path = os.path.join(GOMOKU_RL_DIR, "cfg", "algo", "ppo.yaml")
    cfg = OmegaConf.load(ppo_cfg_path)
    action_spec, _ = _make_specs(board_size, device)
    ac = make_ppo_ac(cfg, action_spec=action_spec, device=device)
    actor  = ac.get_policy_operator()   # observation + action_mask → hidden + probs + action
    critic = ac.get_value_head()        # hidden → state_value
    state_dict = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(state_dict["actor"])
    critic.load_state_dict(state_dict["critic"], strict=False)
    actor.eval()
    critic.eval()
    return GomokuPPOEngine(actor, critic, board_size=board_size, device=device)


class GomokuPPOEngine:
    """
    AlphaZero PolicyValueNetNumpy + MCTSPlayer 역할을 PPO actor/critic으로 대체.
    share_network=True 구조이므로 actor 호출 한 번으로 probs + hidden을 얻고,
    critic에 hidden을 넘겨 state_value(v_s)를 계산한다.

    공개 메서드:
        policy_value_fn(board) → (act_probs_iter, v_s)
            act_probs_iter : zip(positions, probs)  — compute_q_values 호환
            v_s            : critic이 평가한 상태 가치 (scalar)
        get_action(board, temp=1e-3) → int
            MCTSPlayer.get_action 호환 (greedy: argmax(probs))
    """

    def __init__(self, actor, critic, board_size: int = 15, device="cpu"):
        self._actor  = actor
        self._critic = critic
        self._board_size = board_size
        self._device = device

    def _make_td(self, board: GomokuRLBoard) -> TensorDict:
        obs  = board._env.get_encoded_board().to(self._device)  # (1, 3, B, B)
        mask = board._env.get_action_mask().to(self._device)    # (1, B*B) bool
        return TensorDict(
            {"observation": obs, "action_mask": mask},
            batch_size=[1],
            device=self._device,
        )

    def _forward(self, board: GomokuRLBoard):
        """actor 한 번 호출로 probs, hidden, v_s를 모두 반환."""
        td = self._make_td(board)
        with torch.no_grad():
            actor_out = self._actor(td)
            critic_out = self._critic(actor_out.select("hidden"))
        probs = actor_out["probs"][0].cpu().float().numpy()     # (B*B,)
        v_s   = float(critic_out["state_value"][0].item())
        return probs, v_s

    def policy_value_fn(self, board: GomokuRLBoard):
        """compute_q_values()가 기대하는 (iter[(pos, prob)], v_s) 형태로 반환."""
        legal = board.availables
        if not legal:
            return iter([]), 0.0
        probs, v_s = self._forward(board)
        pairs = [(pos, float(probs[pos])) for pos in legal]
        return iter(pairs), v_s

    def get_action(self, board: GomokuRLBoard, temp: float = 1e-3) -> int:
        """확률이 가장 높은 합법 착수를 반환 (greedy). MCTSPlayer.get_action 호환."""
        legal = board.availables
        if not legal:
            return -1
        probs, _ = self._forward(board)
        return max(legal, key=lambda p: probs[p])
