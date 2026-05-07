"""
sessions.py (games.atari.sessions)

세션 디스크 영속화 + 카운터팩추얼 캐시 직렬화 mixin.

`_SessionsMixin` 은 단독으로 인스턴스화하지 않으며, `AtariGame` 이 다중 상속
대상으로 사용한다. 모든 메서드는 `self.X` 로 `AtariGame` 의 상태(episode_data,
analysis_results, counterfactual_cache, session_id, saved_sessions_dir, prefix,
game_id, game_title 등)에 접근한다.

이전 위치(atari_base.py 의 메서드들):
  _session_dir, _list_sessions, _on_save_session, _on_load_session,
  _on_delete_session, _serialize_cf_cache
"""
import os
import copy
import json
import uuid
import shutil
import pickle
from datetime import datetime

from flask_socketio import emit


class _SessionsMixin:
    # ── 디렉토리 / 목록 ──────────────────────────────────────────────────────
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

    # ── CF 캐시 직렬화 ───────────────────────────────────────────────────────
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

    # ── 저장 / 불러오기 / 삭제 (소켓 핸들러) ─────────────────────────────────
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
        sid      = data.get('session_id')
        sess_dir = os.path.join(self._session_dir(), sid)
        if os.path.isdir(sess_dir):
            shutil.rmtree(sess_dir)
        emit(f'{self.prefix}session_deleted', {
            'ok': True, 'message': '기록을 삭제했습니다.',
            'sessions': self._list_sessions(),
        })