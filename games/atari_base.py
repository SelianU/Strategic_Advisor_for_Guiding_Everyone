"""
games/atari_base.py
───────────────────
Atari ALE 게임의 공통 로직을 캡슐화한 베이스 클래스.
새 게임 추가 시 이 클래스를 상속하고 class 속성 + 두 메서드만 구현하면 됩니다.

최소 구현 예시 (games/pong.py):
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
            {'id': 'fire',  'label': 'FIRE', 'actions': [1, 4, 5]},
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

import os, copy, pickle, json, uuid, base64
from abc import ABC, abstractmethod
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np
import gymnasium as gym
from flask_socketio import emit

from llm_feedback import (generate_feedback, DEFAULT_MODEL, FALLBACK_MODELS,
                          FALLBACK_PRIORITY, FALLBACK_POOL, short_model_name)


# ── 공유 유틸리티 ─────────────────────────────────────────────────────────────

def _preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


def encode_frame(frame: np.ndarray) -> str:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')


def encode_frames(frames: list) -> list:
    return [encode_frame(f) for f in frames]


def compose_compare_frame(human_frame: np.ndarray, agent_frame: np.ndarray) -> np.ndarray:
    h, w = human_frame.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = human_frame
    canvas[:, w:] = agent_frame
    cv2.line(canvas, (w, 0), (w, h), (80, 80, 80), 2)
    return canvas


def _hex_to_rgb(hex_color: str) -> str:
    """'#ff8a1c' → '255,138,28'"""
    h = hex_color.lstrip('#')
    return ','.join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ── Grad-CAM (Advantage Stream, DuelingDQN 전용) ─────────────────────────────

class AtariGradCAM:
    """
    DuelingDQN의 Advantage stream에 Grad-CAM을 적용합니다.

    지원 조건: net에 features / grad_rescale / value_stream / advantage_stream 속성 존재.
    torch.compile로 래핑된 경우 _orig_mod를 통해 내부 모델에 접근합니다.

    사용법:
        gcam = AtariGradCAM(net, device)
        heatmap, action, q_values = gcam(stacked_uint8)  # (4,84,84) uint8
        gcam.remove()   # 훅 해제 (반드시 호출)

    반환:
        heatmap  : (84,84) float32 [0,1]
        action   : int (argmax Q-value)
        q_values : np.ndarray (n_actions,)
    """

    def __init__(self, net, device: str):
        import torch.nn as nn
        self.net    = net
        self.device = device
        self._grads = None
        self._feats = None
        self._hooks: list = []
        self._enabled = False

        # torch.compile 래핑 벗기기
        inner = getattr(net, '_orig_mod', net)
        if not all(hasattr(inner, a) for a in ('features', 'grad_rescale',
                                                'value_stream', 'advantage_stream')):
            return  # 지원 불가 모델 → _enabled=False

        self._inner = inner
        last_conv = None
        for m in inner.features.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m

        if last_conv is None:
            return

        def fwd(module, inp, out):
            self._feats = out.detach()

        def bwd(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        self._hooks.append(last_conv.register_forward_hook(fwd))
        self._hooks.append(last_conv.register_full_backward_hook(bwd))
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, state_uint8: np.ndarray):
        """
        state_uint8: (4, 84, 84) uint8
        returns: (heatmap float32 84×84, chosen_action int, q_values ndarray)
        """
        import torch
        import torch.nn.functional as F

        if not self._enabled:
            with torch.no_grad():
                s = torch.from_numpy(state_uint8.astype(np.uint8)).unsqueeze(0).to(self.device)
                q = self.net(s).squeeze(0).cpu().numpy()
            return np.zeros((84, 84), dtype=np.float32), int(np.argmax(q)), q

        inner = self._inner
        inner.zero_grad()

        s = torch.from_numpy(state_uint8.astype(np.float32)).unsqueeze(0).to(self.device)

        # forward — advantage stream 기준
        x        = s.contiguous().float().mul_(1.0 / 255.0)
        features = inner.features(x).flatten(1)
        features = inner.grad_rescale(features)
        value     = inner.value_stream(features)
        advantage = inner.advantage_stream(features)
        q_values  = (value + advantage - advantage.mean(dim=1, keepdim=True))[0]

        chosen = int(q_values.argmax().item())

        # backward on chosen action's advantage
        advantage[0, chosen].backward()

        if self._grads is None or self._feats is None:
            return np.zeros((84, 84), dtype=np.float32), chosen, q_values.detach().cpu().numpy()

        grads   = self._grads[0]   # (C, H, W)
        feats   = self._feats[0]   # (C, H, W)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam     = F.relu((weights * feats).sum(dim=0)).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        heatmap = cv2.resize(cam.astype(np.float32), (84, 84),
                             interpolation=cv2.INTER_LINEAR)

        return heatmap, chosen, q_values.detach().cpu().numpy()


def encode_frame_with_gradcam(
    frame_rgb: np.ndarray,
    heatmap_84: np.ndarray,
    alpha: float = 0.50,
) -> str:
    """
    frame_rgb  : (H, W, 3) uint8 RGB 원본 프레임
    heatmap_84 : (84, 84) float32 [0,1] Grad-CAM heatmap
    alpha      : heatmap 투명도 (기본 0.50)
    반환       : base64 JPEG 문자열
    """
    H, W = frame_rgb.shape[:2]
    hm_scaled = (heatmap_84 * 255).astype(np.uint8)
    hm_color  = cv2.applyColorMap(
        cv2.resize(hm_scaled, (W, H), interpolation=cv2.INTER_LINEAR),
        cv2.COLORMAP_JET,
    )  # BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay   = cv2.addWeighted(frame_bgr, 1.0 - alpha, hm_color, alpha, 0)
    _, buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')


# ── 베이스 클래스 ─────────────────────────────────────────────────────────────

class AtariGame(ABC):
    """
    Atari ALE 게임 베이스 클래스.
    서브클래스는 아래 class 속성과 두 추상 메서드만 구현하면 됩니다.
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

    # ─────────────────────────────────────────────────────────────────────────

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

        self._init()

    def _init(self):
        self._env = gym.make(self.env_name, render_mode='rgb_array',
                             frameskip=1, repeat_action_probability=0.0)
        self._frames = deque(maxlen=4)
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            *self.model_path_parts
        )
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

    # ── 세션 관련 ─────────────────────────────────────────────────────────────

    def _session_dir(self) -> str:
        path = os.path.join(self.saved_sessions_dir, self.game_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _list_sessions(self) -> list:
        game_dir = self._session_dir()
        sessions = []
        for sid in os.listdir(game_dir):
            meta_path = os.path.join(game_dir, sid, 'meta.json')
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                sessions.append({
                    'id': sid,
                    'title': meta.get('title') or '이름 없음',
                    'saved_at': meta.get('saved_at') or '',
                    'summary': meta.get('summary') or '',
                })
            except Exception:
                continue
        sessions.sort(key=lambda x: x['saved_at'], reverse=True)
        return sessions

    # ── 분석 관련 유틸 ─────────────────────────────────────────────────────────

    def _select_top5(self, analyses: list, k: int = 5, min_gap: int = 3) -> list:
        if not analyses:
            return []
        total = len(analyses)
        seg_size = max(1, total // k)
        selected = []
        for seg_idx in range(k):
            start = seg_idx * seg_size
            end = total if seg_idx == k - 1 else (seg_idx + 1) * seg_size
            candidates = sorted(analyses[start:end], key=lambda a: a['loss'], reverse=True)
            for c in candidates:
                if all(abs(c['step'] - s['step']) >= min_gap for s in selected):
                    selected.append(c)
                    break
        return sorted(selected, key=lambda a: a['loss'], reverse=True)

    def _analysis_snapshot(self) -> dict | None:
        if not self.analysis_results:
            return None
        losses = [a['loss'] for a in self.analysis_results]
        p_cnt  = Counter(a['action_name']      for a in self.analysis_results)
        ai_cnt = Counter(a['best_action_name'] for a in self.analysis_results)
        return {
            'avg_loss':      round(float(np.mean(losses)), 3),
            'worst':         max(self.analysis_results, key=lambda a: a['loss']),
            'worst_10':      self._select_top5(self.analysis_results),
            'total_steps':   len(self.analysis_results),
            'agree_rate':    round(sum(1 for a in self.analysis_results if a['is_best']) / len(self.analysis_results) * 100, 1),
            'player_actions': dict(p_cnt),
            'ai_actions':    dict(ai_cnt),
        }

    def _basic_analyze(self) -> dict:
        acts   = [d['action'] for d in self.episode_data if d['action'] is not None]
        total  = len(acts) or 1
        reward = sum(d['reward'] for d in self.episode_data)
        return {'total_reward': int(reward), 'total_steps': total}

    def _serialize_cf_cache(self) -> list:
        saved = []
        for (sid, ei, rh), payload in self.counterfactual_cache.items():
            if sid != self.session_id:
                continue
            item = copy.deepcopy(payload)
            # Grad-CAM 프레임(대용량)은 직렬화에서 제외
            item.pop('gradcam_human', None)
            item.pop('gradcam_agent', None)
            item['entry_index']    = ei
            item['replay_horizon'] = rh
            saved.append(item)
        return saved

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
                theme_rgb    = _hex_to_rgb(game.theme_color),
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

    # ── 게임 루프 ─────────────────────────────────────────────────────────────

    def _on_start(self):
        self.session_id += 1
        self.env_ready         = False
        self.episode_data      = []
        self.valid_entries     = []
        self.analysis_results  = []
        self.counterfactual_cache = {}

        obs_raw, _ = self._env.reset()
        proc = _preprocess(obs_raw)
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
        pre_stacked  = self._get_stacked().copy()
        pre_rgb      = self.last_rgb.copy() if self.last_rgb is not None else self._env.render()

        obs_raw, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        self.last_rgb = obs_raw.copy()
        current_score = float(
            sum(d['reward'] for d in self.episode_data if d.get('action') is not None)
            + float(reward)
        )
        self._frames.append(_preprocess(obs_raw))

        self.episode_data.append({
            'step':            len(self.episode_data),
            'pre_snapshot':    pre_snapshot,
            'pre_stacked_state': pre_stacked,
            'pre_rgb':         pre_rgb,
            'rgb':             obs_raw,
            'stacked_state':   self._get_stacked().copy(),
            'action':          action,
            'reward':          float(reward),
            'done':            done,
        })

        if done:
            self.env_ready = False
            basic = self._basic_analyze()
            emit(f'{self.prefix}over', {**basic, 'has_model': (self.net is not None)})
            if self.net is not None:
                self.socketio.start_background_task(self._run_analysis)
        else:
            emit(f'{self.prefix}frame', {
                'image': encode_frame(obs_raw), 'done': False, 'score': current_score,
            })

    # ── Q-value 분석 (백그라운드) ─────────────────────────────────────────────

    def _run_analysis(self):
        valid = [d for d in self.episode_data if d['action'] is not None]
        if not valid:
            return
        sid   = self.session_id
        total = len(valid)
        self.valid_entries = valid
        self.socketio.emit(f'{self.prefix}analysis_start', {'total': total, 'session_id': sid})

        analyses = []
        for i, entry in enumerate(valid):
            if sid != self.session_id:
                return   # 새 게임 시작됨
            if i % 50 == 0:
                self.socketio.emit(f'{self.prefix}analysis_progress', {
                    'current': i, 'total': total,
                    'pct': round(i / total * 100), 'session_id': sid,
                })
            q_vals   = self._get_q_values(entry['pre_stacked_state'])
            action   = entry['action']
            best_act = int(np.argmax(q_vals))
            player_q = float(q_vals[action])
            best_q   = float(q_vals[best_act])
            analyses.append({
                'step':             entry['step'],
                'action':           action,
                'entry_index':      i,
                'action_name':      self.action_names.get(action,   str(action)),
                'best_action':      best_act,
                'best_action_name': self.action_names.get(best_act, str(best_act)),
                'q_values':         [round(float(v), 3) for v in q_vals],
                'player_q':         round(player_q, 3),
                'best_q':           round(best_q, 3),
                'loss':             round(best_q - player_q, 3),
                'reward':           entry['reward'],
                'is_best':          (action == best_act),
            })

        losses   = [a['loss'] for a in analyses]
        avg_loss = round(float(np.mean(losses)), 3)
        worst    = max(analyses, key=lambda a: a['loss'])
        worst_10 = self._select_top5(analyses)
        agree    = round(sum(1 for a in analyses if a['is_best']) / len(analyses) * 100, 1)
        self.analysis_results = analyses

        p_cnt  = Counter(a['action_name']      for a in analyses)
        ai_cnt = Counter(a['best_action_name'] for a in analyses)

        self.socketio.emit(f'{self.prefix}analysis_done', {
            'avg_loss': avg_loss, 'worst': worst, 'worst_10': worst_10,
            'total_steps': total, 'agree_rate': agree,
            'player_actions': dict(p_cnt), 'ai_actions': dict(ai_cnt),
            'session_id': sid,
        })
        print(f'✅ [{self.game_id}] 분석 완료 avg_loss={avg_loss:.3f} agree={agree}%')

    # ── 카운터팩추얼 ──────────────────────────────────────────────────────────

    def _on_counterfactual(self, data: dict):
        if self.net is None or not self.valid_entries:
            emit(f'{self.prefix}counterfactual_error', {'message': '분석 데이터가 없습니다.'})
            return

        entry_index    = int(data.get('entry_index', -1))
        replay_horizon = int(data.get('horizon', 1200))
        req_sid        = int(data.get('session_id', self.session_id))

        if req_sid != self.session_id:
            emit(f'{self.prefix}counterfactual_error', {
                'message': '이전 판의 요청입니다.',
                'session_id': req_sid, 'entry_index': entry_index,
            })
            return

        cache_key = (req_sid, entry_index, replay_horizon)
        if cache_key in self.counterfactual_cache:
            emit(f'{self.prefix}counterfactual_ready', self.counterfactual_cache[cache_key])
            return

        if not (0 <= entry_index < len(self.valid_entries)):
            emit(f'{self.prefix}counterfactual_error', {
                'message': '후보를 찾지 못했습니다.', 'entry_index': entry_index,
            })
            return

        entry       = self.valid_entries[entry_index]
        q_vals      = self._get_q_values(entry['pre_stacked_state'])
        best_action = int(np.argmax(q_vals))

        # Grad-CAM 인스턴스 (재생 전체에서 공유, 미지원 시 None)
        gcam = self._make_gradcam()

        human_env = self._make_env()
        agent_env = self._make_env()
        try:
            self._restore_env(human_env, entry['pre_snapshot'])
            self._restore_env(agent_env, entry['pre_snapshot'])

            h_stack = deque(entry['pre_stacked_state'], maxlen=4)
            a_stack = deque(entry['pre_stacked_state'], maxlen=4)
            h_score = a_score = 0.0
            h_first = a_first = None
            h_steps: list[int] = []
            a_steps: list[int] = []
            rendered, h_log, a_log = [], [], []
            # Grad-CAM 오버레이 프레임 (인간 / 에이전트 각각)
            gcam_human_frames: list = []
            gcam_agent_frames: list = []
            h_done = a_done = False
            h_frame = a_frame = entry['pre_rgb'].copy()

            for offset in range(replay_horizon):
                h_action = (self.valid_entries[entry_index + offset]['action']
                            if entry_index + offset < len(self.valid_entries) else 0)
                if offset == 0:
                    a_action = best_action
                else:
                    a_action = int(np.argmax(self._get_q_values(np.array(a_stack, dtype=np.uint8))))

                h_log.append(int(h_action))
                a_log.append(int(a_action))

                if not h_done:
                    obs_h, rew_h, t_h, tr_h, _ = human_env.step(h_action)
                    h_frame = obs_h
                    h_stack.append(_preprocess(obs_h))
                    h_score += float(rew_h)
                    if rew_h > 0:
                        h_steps.append(offset + 1)
                        if h_first is None:
                            h_first = offset + 1
                    h_done = t_h or tr_h

                if not a_done:
                    obs_a, rew_a, t_a, tr_a, _ = agent_env.step(a_action)
                    a_frame = obs_a
                    a_stack.append(_preprocess(obs_a))
                    a_score += float(rew_a)
                    if rew_a > 0:
                        a_steps.append(offset + 1)
                        if a_first is None:
                            a_first = offset + 1
                    a_done = t_a or tr_a

                rendered.append(compose_compare_frame(h_frame, a_frame))

                # ── Grad-CAM 오버레이 생성 ──────────────────────────────────
                if gcam is not None:
                    h_hm, _, _ = gcam(np.array(h_stack, dtype=np.uint8))
                    a_hm, _, _ = gcam(np.array(a_stack, dtype=np.uint8))
                    gcam_human_frames.append(encode_frame_with_gradcam(h_frame, h_hm))
                    gcam_agent_frames.append(encode_frame_with_gradcam(a_frame, a_hm))

            gap = float(q_vals[best_action] - q_vals[entry['action']])
            summary = {
                'step':                     entry['step'],
                'loss':                     round(gap, 3),
                'gap':                      round(gap, 4),
                'human_action_name':        self.action_names.get(entry['action'], str(entry['action'])),
                'agent_action_name':        self.action_names.get(best_action,     str(best_action)),
                'human_q':                  round(float(q_vals[entry['action']]), 4),
                'agent_q':                  round(float(q_vals[best_action]), 4),
                'human_score_delta':        round(h_score, 1),
                'agent_score_delta':        round(a_score, 1),
                'human_first_reward_step':  h_first,
                'agent_first_reward_step':  a_first,
                'human_reward_steps':       h_steps,
                'agent_reward_steps':       a_steps,
                'human_done':               h_done,
                'agent_done':               a_done,
                'replay_horizon':           replay_horizon,
                **self._extra_summary(entry),
            }
            feedback, fb_structured, fb_source, fb_model, fb_route = generate_feedback(self.game_id, summary)
            payload = {
                'frames':               encode_frames(rendered),
                'gradcam_human':        gcam_human_frames,   # Grad-CAM 오버레이 (인간)
                'gradcam_agent':        gcam_agent_frames,   # Grad-CAM 오버레이 (에이전트)
                'has_gradcam':          gcam is not None,
                'human_actions':        h_log,
                'agent_actions':        a_log,
                'summary':              summary,
                'feedback':             feedback,
                'feedback_structured':  fb_structured,
                'feedback_source':      fb_source,
                'feedback_model':       fb_model,
                'feedback_route':       fb_route,
                'session_id':           req_sid,
                'entry_index':          entry_index,
            }
        finally:
            human_env.close()
            agent_env.close()
            if gcam is not None:
                gcam.remove()

        self.counterfactual_cache[cache_key] = payload
        emit(f'{self.prefix}counterfactual_ready', payload)

    # ── 세션 저장 / 불러오기 / 삭제 ──────────────────────────────────────────

    def _on_save_session(self, data: dict):
        if not self.episode_data or not self.analysis_results:
            emit(f'{self.prefix}session_saved', {'ok': False, 'message': '저장할 분석 결과가 없습니다.'})
            return
        title    = (data.get('title') or '').strip() or f'{self.game_title} 기록'
        sid      = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
        sess_dir = os.path.join(self._session_dir(), sid)
        os.makedirs(sess_dir, exist_ok=True)
        basic    = self._basic_analyze()
        snapshot = self._analysis_snapshot()
        meta = {
            'game':       self.game_id,
            'title':      title,
            'saved_at':   datetime.now().isoformat(timespec='seconds'),
            'summary':    f"점수 {basic['total_reward']} · 스텝 {basic['total_steps']}",
            'basic_report': basic,
        }
        with open(os.path.join(sess_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(sess_dir, 'session.pkl'), 'wb') as f:
            pickle.dump({
                'episode_data':        self.episode_data,
                'analysis_results':    self.analysis_results,
                'analysis_snapshot':   snapshot,
                'cached_counterfactuals': self._serialize_cf_cache(),
            }, f)
        emit(f'{self.prefix}session_saved', {
            'ok': True, 'message': '기록을 저장했습니다.',
            'sessions': self._list_sessions(),
        })

    def _on_load_session(self, data: dict):
        sid      = data.get('session_id')
        sess_dir = os.path.join(self._session_dir(), sid)
        meta_p   = os.path.join(sess_dir, 'meta.json')
        data_p   = os.path.join(sess_dir, 'session.pkl')
        if not os.path.isfile(meta_p) or not os.path.isfile(data_p):
            emit(f'{self.prefix}session_loaded', {'ok': False, 'message': '기록을 찾지 못했습니다.'})
            return
        with open(meta_p, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        with open(data_p, 'rb') as f:
            payload = pickle.load(f)

        self.session_id += 1
        self.episode_data     = payload['episode_data']
        self.analysis_results = payload['analysis_results']
        self.valid_entries    = [e for e in self.episode_data if e.get('action') is not None]
        self.counterfactual_cache = {}
        cached = payload.get('cached_counterfactuals') or []
        for item in cached:
            ei = int(item.get('entry_index', -1))
            rh = int(item.get('replay_horizon', 1200))
            restored = copy.deepcopy(item)
            restored['session_id'] = self.session_id
            self.counterfactual_cache[(self.session_id, ei, rh)] = restored

        emit(f'{self.prefix}session_loaded', {
            'ok':                    True,
            'meta':                  meta,
            'basic':                 meta.get('basic_report', {}),
            'analysis':              payload.get('analysis_snapshot') or self._analysis_snapshot(),
            'cached_counterfactuals': cached,
            'session_id':            self.session_id,
        })

    def _on_delete_session(self, data: dict):
        import shutil
        sid      = data.get('session_id')
        sess_dir = os.path.join(self._session_dir(), sid)
        if os.path.isdir(sess_dir):
            shutil.rmtree(sess_dir)
        emit(f'{self.prefix}session_deleted', {
            'ok': True, 'message': '기록을 삭제했습니다.',
            'sessions': self._list_sessions(),
        })