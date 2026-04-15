import sys
import os
import copy
import numpy as np
import torch
from tensordict import TensorDict
try:
    from torchrl.data import Categorical as _ActionSpec
    from torchrl.data import Composite as _CompositeSpec
    from torchrl.data import Unbounded as _ObsSpec
    from torchrl.data import Binary as _MaskSpec
except ImportError:
    from torchrl.data.tensor_specs import DiscreteTensorSpec as _ActionSpec
    from torchrl.data.tensor_specs import CompositeSpec as _CompositeSpec
    from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec as _ObsSpec
    from torchrl.data.tensor_specs import BinaryDiscreteTensorSpec as _MaskSpec
from omegaconf import OmegaConf

GOMOKU_RL_DIR = os.path.join(os.path.dirname(__file__), "gomoku_rl")
if GOMOKU_RL_DIR not in sys.path:
    sys.path.insert(0, GOMOKU_RL_DIR)

from gomoku_rl.core import Gomoku
from gomoku_rl.policy import get_policy


def _make_specs(board_size: int, device):
    n_actions = board_size * board_size
    action_spec = _ActionSpec(n_actions, shape=[1], device=device)
    observation_spec = _CompositeSpec(
        {
            "observation": _ObsSpec(device=device, shape=[1, 3, board_size, board_size]),
            "action_mask": _MaskSpec(n=n_actions, device=device, shape=[1, n_actions], dtype=torch.bool),
        },
        shape=[1],
        device=device,
    )
    return action_spec, observation_spec


class GomokuRLBoard:
    def __init__(self, board_size: int = 15, n_in_row: int = 5, device="cpu"):
        self.width = board_size
        self.height = board_size
        self.n_in_row = n_in_row
        self._device = device
        self._env = Gomoku(num_envs=1, board_size=board_size, device=device)
        self._env.reset()
        self.states: dict = {}

    def init_board(self, start_player: int = 0):
        self._env.reset()
        self.states = {}

    @property
    def current_player(self) -> int:
        return 1 if self._env.turn[0].item() == 0 else 2

    @property
    def availables(self) -> list:
        return [i for i, v in enumerate(self._env.get_action_mask()[0].tolist()) if v]

    @property
    def last_move(self) -> int:
        return int(self._env.last_move[0].item())

    def do_move(self, move: int):
        player = self.current_player
        self._env.step(torch.tensor([move], device=self._device))
        self.states[move] = player

    def game_end(self):
        done = self._env.done[0].item()
        if not done:
            return False, -1
        if self._env.move_count[0].item() >= self.width * self.height:
            return True, -1
        current_turn = self._env.turn[0].item()
        winner = 1 if current_turn == 1 else 2
        return True, winner

    def __deepcopy__(self, memo):
        new_obj = GomokuRLBoard.__new__(GomokuRLBoard)
        new_obj.width = self.width
        new_obj.height = self.height
        new_obj.n_in_row = self.n_in_row
        new_obj._device = self._device
        new_obj._env = Gomoku(num_envs=1, board_size=self.width, device=self._device)
        new_obj._env.board = self._env.board.clone()
        new_obj._env.done = self._env.done.clone()
        new_obj._env.turn = self._env.turn.clone()
        new_obj._env.move_count = self._env.move_count.clone()
        new_obj._env.last_move = self._env.last_move.clone()
        new_obj.states = dict(self.states)
        return new_obj


def load_gomoku_ppo(checkpoint_path: str, board_size: int = 15, device="cpu"):
    from gomoku_rl.policy.common import make_ppo_ac
    ppo_cfg_path = os.path.join(GOMOKU_RL_DIR, "cfg", "algo", "ppo.yaml")
    cfg = OmegaConf.load(ppo_cfg_path)
    action_spec, _ = _make_specs(board_size, device)
    ac = make_ppo_ac(cfg, action_spec=action_spec, device=device)
    actor = ac.get_policy_operator()
    critic = ac.get_value_head()
    state_dict = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(state_dict["actor"])
    critic.load_state_dict(state_dict["critic"], strict=False)
    actor.eval()
    critic.eval()
    return GomokuPPOEngine(actor, critic, board_size=board_size, device=device)


class GomokuPPOEngine:
    def __init__(self, actor, critic, board_size: int = 15, device="cpu"):
        self._actor = actor
        self._critic = critic
        self._board_size = board_size
        self._device = device

    def _make_td(self, board: GomokuRLBoard) -> TensorDict:
        obs = board._env.get_encoded_board().to(self._device)
        mask = board._env.get_action_mask().to(self._device)
        return TensorDict(
            {"observation": obs, "action_mask": mask},
            batch_size=[1],
            device=self._device,
        )

    def _forward(self, board: GomokuRLBoard):
        td = self._make_td(board)
        with torch.no_grad():
            actor_out = self._actor(td)
            critic_out = self._critic(actor_out.select("hidden"))
        probs = actor_out["probs"][0].cpu().float().numpy()
        v_s = float(critic_out["state_value"][0].item())
        return probs, v_s

    def policy_value_fn(self, board: GomokuRLBoard):
        legal = board.availables
        if not legal:
            return iter([]), 0.0
        probs, v_s = self._forward(board)
        return iter([(pos, float(probs[pos])) for pos in legal]), v_s

    def get_action(self, board: GomokuRLBoard, temp: float = 1e-3) -> int:
        legal = board.availables
        if not legal:
            return -1
        probs, _ = self._forward(board)
        return max(legal, key=lambda p: probs[p])
