"""
base.py (games.atari.base)

Atari ALE 게임의 공통 로직을 캡슐화한 베이스 클래스.

리팩토링으로 다음과 같이 분리되었다:
  - games/atari/preprocessing.py   : 프레임 전처리 / 인코딩 유틸
  - games/atari/gradcam.py         : AtariGradCAM + encode_frame_with_gradcam
  - games/atari/sessions.py        : _SessionsMixin (세션 디스크 영속화)
  - games/atari/analysis.py        : _AnalysisMixin (Q-value 분석)
  - games/atari/counterfactual.py  : _CounterfactualMixin (CF 리플레이)
  - games/atari/base.py (본 파일)  : AtariGame (mixin 결합 + 게임 루프 + 추상 메서드)

서브클래스(SpaceInvadersGame 등)는 이전과 동일하게 `class XGame(AtariGame):` 만
유지하면 된다. 메서드 시그니처는 모두 그대로이고, mixin 분리는 내부 구현 디테일.

최소 구현 예시 (games/pong.py):
    from games.atari import AtariGame      # ← 변경된 import 경로 (구: games.atari_base)

    class PongGame(AtariGame):
        game_id    = 'pong'
        game_title = 'PONG'
        game_icon  = '🏓'
        env_name   = 'ALE/Pong-v5'
        prefix     = 'po_'
        theme_color = '#00f5ff'
        model_path_parts = ('ai_agents', 'pong', 'checkpoints', 'best_model.pth')
        action_names = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}
        keyboard_keys = [
            {'id': 'left',  'label': '←',    'actions': [3, 5]},
            {'id': 'right', 'label': '→',    'actions': [2, 4]},
            {'id': 'fire',  'label': 'SPACE', 'actions': [1, 4, 5]},
        ]
        key_combos = {'left+fire': 5, 'right+fire': 4, 'left': 3, 'right': 2, 'fire': 1, '': 0}

        def _load_model(self, path):
            if not os.path.exists(path): return None
            from ai_agents.pong import load_d3qn
            net, _ = load_d3qn(path, self.device)
            return net

        def _get_q_values(self, stacked_state):
            from ai_agents.pong import get_q_values
            return get_q_values(self.net, stacked_state, self.device)
"""
import os
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import gymnasium as gym
from flask_socketio import emit

from llm_feedback import (DEFAULT_MODEL,
                          FALLBACK_PRIORITY, FALLBACK_POOL, short_model_name)

from .preprocessing  import preprocess, encode_frame, hex_to_rgb
from .gradcam        import AtariGradCAM
from .sessions       import _SessionsMixin
from .analysis       import _AnalysisMixin
from .counterfactual import _CounterfactualMixin
from .practice       import _PracticeMixin


# ── 프로젝트 루트 (모델 체크포인트 경로 계산용) ──────────────────────────────
# 이 파일은 <project>/games/atari/base.py 에 있으므로 dirname 을 3번 적용.
# 기존 atari_base.py 위치(<project>/games/atari_base.py) 대비 한 단계 깊어졌으니
# dirname 횟수를 2 → 3 으로 늘려 보정.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── 베이스 클래스 ────────────────────────────────────────────────────────────
class AtariGame(_SessionsMixin, _AnalysisMixin, _CounterfactualMixin, _PracticeMixin, ABC):
    """
    Atari ALE 게임 베이스 클래스.
    서브클래스는 아래 class 속성과 두 추상 메서드만 구현하면 됩니다.

    Mixin 으로부터 상속되는 메서드(서브클래스에서 그대로 쓰거나 override 가능):
      _SessionsMixin       : _session_dir, _list_sessions,
                             _on_save_session, _on_load_session, _on_delete_session,
                             _serialize_cf_cache
      _AnalysisMixin       : _select_top5, _analysis_snapshot,
                             _basic_analyze, _run_analysis
      _CounterfactualMixin : _on_counterfactual
    """

    # ── 필수 override 항목 ────────────────────────────────────────────────────
    game_id: str    = ''           # 세션 폴더명 / LLM 피드백 game_type
    game_title: str = ''           # 화면 표시용 제목
    game_icon: str  = '🎮'
    env_name: str   = ''           # gymnasium 환경 ID
    prefix: str     = ''           # 소켓 이벤트 접두사 ('bo_', 'si_' …)
    theme_color: str = '#39ff14'   # 헥스 컬러 (CSS 테마)

    # 모델 체크포인트 경로 (프로젝트 루트 기준 경로 파트 튜플)
    model_path_parts: tuple = ()

    # 학습 시 사용한 frame skip — 에이전트 분석/카운터팩추얼에 사용
    frame_skip: int = 4
    # 인간 플레이 시 frame skip (에이전트 분석용 frame_skip 과 별개)
    play_frame_skip: int = 1

    # 액션 이름 {int: str}
    action_names: dict = {}

    # 가상 키보드: 컨트롤팩추얼 리플레이에 표시되는 키 목록
    # 각 항목: {'id': str, 'label': str, 'actions': [int, ...]}
    # actions 리스트에 해당 액션이 포함되면 해당 키가 강조됨
    keyboard_keys: list = []

    # 키→액션 매핑: 누른 키 조합 문자열 → 액션 int
    # 키 조합은 'left', 'right', 'fire'를 '+' 로 연결 (알파벳 순)
    # 예: {'left+fire': 5, 'right+fire': 4, 'left': 3, 'right': 2, 'fire': 1, '': 0}
    key_combos: dict = {}

    # URL 경로 (기본값: /{game_id}, 필요 시 서브클래스에서 override)
    @property
    def url_path(self) -> str:
        return f'/{self.game_id}'

    # ── 초기화 ────────────────────────────────────────────────────────────────
    def __init__(self, device: str, socketio, app, saved_sessions_dir: str):
        self.device = device
        self.socketio = socketio
        self.app = app
        self.saved_sessions_dir = saved_sessions_dir

        # 에피소드 상태
        self._env = None
        self._frames: deque = deque(maxlen=4)
        self.net = None

        self.episode_data: list = []
        self.env_ready: bool = False
        self.last_rgb = None
        self.session_id: int = 0
        self.valid_entries: list = []
        self.analysis_results: list = []
        self.counterfactual_cache: dict = {}
        self._practice_mode: bool = False

        self._init()

    def _init(self):
        self._env = gym.make(self.env_name, render_mode='rgb_array',
                             frameskip=1, repeat_action_probability=0.0)
        self._frames = deque(maxlen=4)
        model_path = os.path.join(_PROJECT_ROOT, *self.model_path_parts)
        self.net = self._load_model(model_path)
        if self.net is None:
            print(f'⚠  [{self.game_id}] 모델 없음: {model_path}')
        else:
            print(f'✅ [{self.game_id}] 모델 로드 완료')
        self._register_route()
        self._register_handlers()

    # ── 추상 메서드 (서브클래스 필수 구현) ────────────────────────────────────
    @abstractmethod
    def _load_model(self, path: str):
        """모델 파일 로드. 파일 없으면 None 반환."""
        ...

    @abstractmethod
    def _get_q_values(self, stacked_state: np.ndarray) -> np.ndarray:
        """4-프레임 스택 상태로 Q-value 배열 반환."""
        ...

    def _extra_summary(self, entry: dict) -> dict:
        """게임별 추가 summary 필드. 서브클래스에서 필요 시 override."""
        return {}

    def _extra_frame_data(self, obs_raw: np.ndarray) -> dict:
        """매 프레임 emit에 포함할 게임별 추가 데이터. 서브클래스에서 필요 시 override."""
        return {}

    def _compute_achievements(self) -> list:
        """게임 종료 후 도전과제 달성 여부 반환. 서브클래스에서 override."""
        return []

    def _achievement_report(self, achieved=None) -> list:
        """전체 도전과제 목록에 현재 달성 상태를 합쳐 반환."""
        all_achievements = getattr(self, 'achievements', []) or []
        achieved_ids = set(getattr(self, '_ach_unlocked', set()))
        for item in achieved or []:
            ach_id = item.get('id') if isinstance(item, dict) else None
            if ach_id:
                achieved_ids.add(ach_id)

        report = []
        for ach in all_achievements:
            item = dict(ach)
            item['unlocked'] = item.get('id') in achieved_ids
            report.append(item)
        return report

    def _on_episode_reset(self):
        """게임 시작 시 게임별 상태 초기화 훅. 서브클래스에서 override."""
        pass

    def _check_realtime_achievements(self, entry: dict) -> list:
        """매 스텝 후 새로 달성된 도전과제 반환. 서브클래스에서 override."""
        return []

    # ── Grad-CAM 헬퍼 ────────────────────────────────────────────────────────
    # self.net이 DuelingDQN 구조일 때 자동으로 활성화됩니다.
    # 다른 아키텍처를 쓰는 게임은 이 메서드를 override하세요.
    def _make_gradcam(self) -> 'AtariGradCAM | None':
        """Grad-CAM 인스턴스 생성. 지원 불가 모델이면 None 반환."""
        if self.net is None:
            return None
        gcam = AtariGradCAM(self.net, self.device)
        return gcam if gcam.enabled else None

    # ── 환경 관련 ─────────────────────────────────────────────────────────────
    def _make_env(self):
        return gym.make(self.env_name, render_mode='rgb_array',
                        frameskip=1, repeat_action_probability=0.0)

    def _restore_env(self, env, snapshot):
        env.reset()
        env.unwrapped.ale.restoreSystemState(snapshot)

    def _get_stacked(self) -> np.ndarray:
        return np.array(self._frames, dtype=np.uint8)

    # ── Flask 라우트 등록 ──────────────────────────────────────────────────────
    def _register_route(self):
        game = self

        def page_view():
            from flask import render_template
            return render_template(
                'atari_game.html',
                game_id      = game.game_id,
                game_title   = game.game_title,
                game_icon    = game.game_icon,
                prefix       = game.prefix,
                theme_color  = game.theme_color,
                theme_rgb    = hex_to_rgb(game.theme_color),
                has_model    = (game.net is not None),
                action_names = game.action_names,
                keyboard_keys = game.keyboard_keys,
                key_combos   = game.key_combos,
                has_openrouter   = bool(os.getenv('OPENROUTER_API_KEY')),
                openrouter_model = os.getenv('OPENROUTER_MODEL', DEFAULT_MODEL),
                primary_short    = short_model_name(os.getenv('OPENROUTER_MODEL', DEFAULT_MODEL)),
                priority_models  = [(m, short_model_name(m)) for m in FALLBACK_PRIORITY],
                pool_models      = [(m, short_model_name(m)) for m in FALLBACK_POOL],
                saved_sessions   = game._list_sessions(),
                game_info        = getattr(game, 'game_info', {}),
                achievements     = getattr(game, 'achievements', []),
            )

        self.app.add_url_rule(
            self.url_path,
            endpoint=f'{self.game_id}_page',
            view_func=page_view,
        )

    # ── 소켓 핸들러 등록 ──────────────────────────────────────────────────────
    def _register_handlers(self):
        sio = self.socketio
        p   = self.prefix
        g   = self

        sio.on_event(f'{p}start',                 lambda:   g._on_start())
        sio.on_event(f'{p}action',                lambda d: g._on_action(d))
        sio.on_event(f'{p}list_sessions',         lambda:   emit(f'{p}sessions_list', {'sessions': g._list_sessions()}))
        sio.on_event(f'{p}save_session',          lambda d: g._on_save_session(d))
        sio.on_event(f'{p}load_session',          lambda d: g._on_load_session(d))
        sio.on_event(f'{p}delete_session',        lambda d: g._on_delete_session(d))
        sio.on_event(f'{p}request_counterfactual', lambda d: g._on_counterfactual(d))
        sio.on_event(f'{p}start_practice',         lambda d: g._on_start_practice(d))
        sio.on_event(f'{p}practice_action',        lambda d: g._on_practice_action(d))
        sio.on_event(f'{p}stop_practice',          lambda:   g._on_stop_practice())

    # ── 게임 루프 ─────────────────────────────────────────────────────────────
    def _on_start(self):
        self.session_id += 1
        self.env_ready         = False
        self.episode_data      = []
        self.valid_entries     = []
        self.analysis_results  = []
        self.counterfactual_cache = {}
        self._ach_unlocked: set = set()
        self._on_episode_reset()

        obs_raw, _ = self._env.reset()
        proc = preprocess(obs_raw)
        self._frames.clear()
        for _ in range(4):
            self._frames.append(proc)
        self.last_rgb = obs_raw.copy()

        self.episode_data.append({
            'step': 0, 'rgb': obs_raw,
            'stacked_state': self._get_stacked().copy(),
            'action': None, 'reward': 0.0, 'done': False,
        })
        self.env_ready = True
        emit(f'{self.prefix}frame', {
            'image': encode_frame(obs_raw), 'done': False,
            'score': 0.0, 'session_id': self.session_id,
        })

    def _on_action(self, data: dict):
        if not self.env_ready:
            return
        action       = int(data.get('action', 0))
        pre_snapshot = self._env.unwrapped.ale.cloneSystemState()
        pre_ram      = self._env.unwrapped.ale.getRAM().copy()
        pre_stacked  = self._get_stacked().copy()
        pre_rgb      = self.last_rgb.copy() if self.last_rgb is not None else self._env.render()

        n_skip = self.play_frame_skip
        total_reward = 0.0
        terminated = truncated = False
        obs_raw = None
        frame_buf: list = []
        step_info: dict = {}
        for i in range(n_skip):
            obs_raw, rew, terminated, truncated, step_info = self._env.step(action)
            total_reward += rew
            if i >= n_skip - 2:
                frame_buf.append(obs_raw)
            if terminated or truncated:
                break
        obs_for_stack = (np.maximum(frame_buf[0], frame_buf[1])
                         if len(frame_buf) == 2 else obs_raw)
        reward = total_reward
        done = terminated or truncated
        self.last_rgb = obs_raw.copy()
        current_score = float(
            sum(d['reward'] for d in self.episode_data if d.get('action') is not None)
            + float(reward)
        )
        self._frames.append(preprocess(obs_for_stack))

        self.episode_data.append({
            'step':            len(self.episode_data),
            'pre_snapshot':    pre_snapshot,
            'pre_ram':         pre_ram,
            'pre_stacked_state': pre_stacked,
            'pre_rgb':         pre_rgb,
            'rgb':             obs_raw,
            'stacked_state':   self._get_stacked().copy(),
            'action':          action,
            'reward':          float(reward),
            'done':            done,
            'lives':           step_info.get('lives'),
        })

        new_achs = self._check_realtime_achievements(self.episode_data[-1])
        for ach in new_achs:
            emit(f'{self.prefix}achievement', ach)

        if done:
            self.env_ready = False
            basic = self._basic_analyze()
            achievements = self._achievement_report(self._compute_achievements())
            emit(f'{self.prefix}over', {**basic, 'has_model': (self.net is not None),
                                        'achievements': achievements})
            if self.net is not None:
                self.socketio.start_background_task(self._run_analysis)
        else:
            emit(f'{self.prefix}frame', {
                'image': encode_frame(obs_raw), 'done': False, 'score': current_score,
                **self._extra_frame_data(obs_raw),
            })
