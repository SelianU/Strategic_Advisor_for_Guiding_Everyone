"""
practice.py (games.atari.practice)
연습 모드 mixin — Top-Moment 스냅샷에서 인간이 재플레이하는 기능.
"""
import numpy as np
from collections import deque, Counter
from flask_socketio import emit

from .preprocessing import preprocess, encode_frame


class _PracticeMixin:

    def _on_start_practice(self, data: dict):
        """해당 entry의 pre_snapshot으로 env를 복원하고 연습 모드 시작."""
        try:
            entry_index = int(data.get('entry_index', -1))
            horizon     = int(data.get('horizon', 1200))
        except (TypeError, ValueError) as e:
            emit(f'{self.prefix}practice_error', {'message': f'잘못된 파라미터: {e}'})
            return

        if not (0 <= entry_index < len(self.valid_entries)):
            emit(f'{self.prefix}practice_error', {'message': '유효하지 않은 엔트리'})
            return

        try:
            entry = self.valid_entries[entry_index]
            self._restore_env(self._env, entry['pre_snapshot'])
        except Exception as e:
            print(f'[practice] env 복원 실패: {e}')
            emit(f'{self.prefix}practice_error', {'message': f'환경 복원 실패: {e}'})
            return

        self._practice_mode        = True
        self._practice_horizon     = horizon
        self._practice_steps       = 0
        self._practice_entry_index = entry_index
        self._practice_actions     = []
        self._practice_done_sent   = False
        self._practice_stacked     = deque(entry['pre_stacked_state'], maxlen=4)

        emit(f'{self.prefix}practice_ready', {
            'entry_index': entry_index,
            'first_frame': encode_frame(entry['pre_rgb']),
        })

    def _on_practice_action(self, data: dict):
        """연습 중 인간 액션 처리. 게임 frame 전송."""
        if not getattr(self, '_practice_mode', False):
            return
        try:
            action = int(data.get('action', 0))
        except (TypeError, ValueError):
            action = 0
        n_skip = self.play_frame_skip
        total_reward = 0.0
        terminated = truncated = False
        obs_raw = None
        frame_buf = []

        try:
            for i in range(n_skip):
                obs_raw, rew, terminated, truncated, _ = self._env.step(action)
                total_reward += rew
                if i >= n_skip - 2:
                    frame_buf.append(obs_raw)
                if terminated or truncated:
                    break
        except Exception as e:
            print(f'[practice] env.step 실패: {e}')
            self._practice_mode = False
            emit(f'{self.prefix}practice_error', {'message': f'게임 스텝 실패: {e}'})
            return

        obs_for_stack = (np.maximum(frame_buf[0], frame_buf[1])
                         if len(frame_buf) == 2 else obs_raw)
        self._practice_stacked.append(preprocess(obs_for_stack))
        self._practice_steps += 1

        # Q-value 체크
        if self.net is not None:
            q_vals   = self._get_q_values(np.array(self._practice_stacked, dtype=np.uint8))
            best_act = int(np.argmax(q_vals))
        else:
            best_act = action

        self._practice_actions.append({
            'action':           action,
            'action_name':      self.action_names.get(action, str(action)),
            'best_action':      best_act,
            'best_action_name': self.action_names.get(best_act, str(best_act)),
            'reward':           total_reward,
            'is_best':          (action == best_act),
        })

        done = terminated or truncated or (self._practice_steps >= self._practice_horizon)

        emit(f'{self.prefix}practice_frame', {
            'image': encode_frame(obs_raw),
            'score': round(sum(a['reward'] for a in self._practice_actions), 1),
            'done':  done,
        })

        if done:
            self._finish_practice()

    def _on_stop_practice(self):
        """클라이언트 30초 타이머 종료 시 현재까지의 연습 결과를 확정."""
        if getattr(self, '_practice_mode', False) or hasattr(self, '_practice_actions'):
            self._finish_practice()

    def _practice_original_stats(self):
        """연습 기준 entry까지 원본 플레이의 누적 통계를 계산."""
        orig = {}
        entry_index = getattr(self, '_practice_entry_index', -1)
        if self.analysis_results and entry_index >= 0:
            sub = self.analysis_results[:entry_index + 1]
            if sub:
                orig['agree_rate']     = round(sum(1 for a in sub if a['is_best']) / len(sub) * 100, 1)
                orig['avg_loss']       = round(float(np.mean([a['loss'] for a in sub])), 3)
                orig['player_actions'] = dict(Counter(a['action_name']      for a in sub))
                orig['ai_actions']     = dict(Counter(a['best_action_name'] for a in sub))
        return orig

    def _finish_practice(self):
        """연습 종료 후 통계 계산 및 결과 전송."""
        if getattr(self, '_practice_done_sent', False):
            return
        self._practice_done_sent = True
        self._practice_mode = False
        acts = getattr(self, '_practice_actions', [])
        orig = self._practice_original_stats()
        if not acts:
            emit(f'{self.prefix}practice_done', {
                'agree_rate':     0,
                'score':          0,
                'total_steps':    0,
                'player_actions': {},
                'ai_actions':     {},
                'original':       orig,
                'improved':       False,
                'entry_index':    getattr(self, '_practice_entry_index', None),
            })
            return

        total      = len(acts)
        agree_rate = round(sum(1 for a in acts if a['is_best']) / total * 100, 1)
        score      = round(sum(a['reward'] for a in acts), 1)
        p_cnt      = dict(Counter(a['action_name']      for a in acts))
        ai_cnt     = dict(Counter(a['best_action_name'] for a in acts))

        improved = agree_rate > orig.get('agree_rate', 0)

        emit(f'{self.prefix}practice_done', {
            'agree_rate':     agree_rate,
            'score':          score,
            'total_steps':    total,
            'player_actions': p_cnt,
            'ai_actions':     ai_cnt,
            'original':       orig,
            'improved':       improved,
            'entry_index':    self._practice_entry_index,
        })
