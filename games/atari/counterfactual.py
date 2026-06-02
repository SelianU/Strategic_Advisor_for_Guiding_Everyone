"""
counterfactual.py (games.atari.counterfactual)

카운터팩추얼 리플레이 생성 mixin.

선택된 valid entry 시점에서:
  1) 인간 환경 — 실제 진행한 액션 시퀀스대로 재생
  2) 에이전트 환경 — 첫 액션을 best_action 으로, 이후엔 매 스텝 best 선택
양쪽 환경을 `replay_horizon` 스텝까지 진행하며 좌우로 합친 비교 프레임을 생성하고,
DuelingDQN 의 경우 Grad-CAM 오버레이 프레임도 만든다.

이전 위치: atari_base.py 의 _on_counterfactual 메서드 한 덩어리.
`AtariGame` 의 다중 상속 부모로만 사용된다.
"""
import copy
from collections import deque

import numpy as np
from flask_socketio import emit

from llm_feedback import generate_feedback

from .preprocessing import preprocess, encode_frames, compose_compare_frame
from .gradcam       import encode_frame_with_gradcam


class _CounterfactualMixin:

    def _on_counterfactual(self, data: dict):
        if self.net is None or not self.valid_entries:
            emit(f'{self.prefix}counterfactual_error', {'message': '분석 데이터가 없습니다.'})
            return

        entry_index     = int(data.get('entry_index', -1))
        horizon_frames  = int(data.get('horizon', 1800))   # 렌더 프레임 수 기준
        # 에이전트 decision step 수로 변환 (frame_skip당 1스텝)
        replay_horizon  = max(1, horizon_frames // self.frame_skip)
        req_sid         = int(data.get('session_id', self.session_id))

        if req_sid != self.session_id:
            emit(f'{self.prefix}counterfactual_error', {
                'message': '이전 판의 요청입니다.',
                'session_id': req_sid, 'entry_index': entry_index,
            })
            return

        cache_key = (req_sid, entry_index, horizon_frames)
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

            # 에이전트 결정 1번 = frame_skip 프레임
            # 인간은 play_frame_skip 단위로 플레이했으므로,
            # 에이전트 결정 1번 동안 소비되는 인간 entry 수 = frame_skip // play_frame_skip
            n_h = self.frame_skip // max(1, self.play_frame_skip)

            for agent_step in range(replay_horizon):
                if agent_step == 0:
                    a_action = best_action
                else:
                    a_action = int(np.argmax(self._get_q_values(np.array(a_stack, dtype=np.uint8))))

                # ── 인간: n_h개 entry를 각 play_frame_skip 프레임씩 재생 ──────
                h_sub: list = []
                h_log_chunk: list = []
                if not h_done:
                    h_total_rew = 0.0
                    h_buf_all: list = []
                    for k in range(n_h):
                        h_idx = entry_index + agent_step * n_h + k
                        h_action_k = (self.valid_entries[h_idx]['action']
                                      if h_idx < len(self.valid_entries) else 0)
                        for _i in range(self.play_frame_skip):
                            obs_h, rew_h, t_h, tr_h, _ = human_env.step(h_action_k)
                            h_total_rew += rew_h
                            h_sub.append(obs_h)
                            h_buf_all.append(obs_h)
                            if t_h or tr_h:
                                h_done = True
                                break
                        h_log_chunk.extend([int(h_action_k)] * self.play_frame_skip)
                        if h_done:
                            break
                    if h_buf_all:
                        h_frame = h_buf_all[-1]
                        obs_h_stack = (np.maximum(h_buf_all[-2], h_buf_all[-1])
                                       if len(h_buf_all) >= 2 else h_frame)
                        h_stack.append(preprocess(obs_h_stack))
                    h_score += h_total_rew
                    if h_total_rew > 0:
                        h_steps.append(agent_step + 1)
                        if h_first is None:
                            h_first = agent_step + 1

                # ── 에이전트: action 1개를 frame_skip 프레임 유지 ─────────────
                a_sub: list = []
                if not a_done:
                    a_total_rew = 0.0
                    a_buf: list = []
                    for _i in range(self.frame_skip):
                        obs_a, rew_a, t_a, tr_a, _ = agent_env.step(a_action)
                        a_total_rew += rew_a
                        a_sub.append(obs_a)
                        if _i >= self.frame_skip - 2:
                            a_buf.append(obs_a)
                        if t_a or tr_a:
                            break
                    a_frame = a_sub[-1]
                    obs_a_stack = np.maximum(a_buf[0], a_buf[1]) if len(a_buf) == 2 else a_frame
                    a_stack.append(preprocess(obs_a_stack))
                    a_score += a_total_rew
                    if a_total_rew > 0:
                        a_steps.append(agent_step + 1)
                        if a_first is None:
                            a_first = agent_step + 1
                    a_done = t_a or tr_a

                # ── 프레임 수 동기화 및 렌더링 ────────────────────────────────
                n_sub = max(len(h_sub), len(a_sub))
                h_sub      += [h_frame] * (n_sub - len(h_sub))
                a_sub      += [a_frame] * (n_sub - len(a_sub))
                h_log_chunk += [0]       * (n_sub - len(h_log_chunk))

                h_log.extend(h_log_chunk)
                a_log.extend([int(a_action)] * n_sub)

                if gcam is not None:
                    h_hm, _, _ = gcam(np.array(h_stack, dtype=np.uint8))
                    a_hm, _, _ = gcam(np.array(a_stack, dtype=np.uint8))

                for hf, af in zip(h_sub, a_sub):
                    rendered.append(compose_compare_frame(hf, af))
                    if gcam is not None:
                        gcam_human_frames.append(encode_frame_with_gradcam(hf, h_hm))
                        gcam_agent_frames.append(encode_frame_with_gradcam(af, a_hm))

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
            # eventlet 이벤트 루프에 제어권을 넘겨 네트워크 I/O 준비
            try:
                import eventlet as _ev
                _ev.sleep(0)
            except ImportError:
                pass
            feedback, fb_structured, fb_source, fb_model, fb_route = generate_feedback(self.game_id, summary)

            # entry_index까지의 누적 통계
            cumul = {}
            if self.analysis_results:
                sub = self.analysis_results[:entry_index + 1]
                if sub:
                    cumul['agree_rate']     = round(sum(1 for a in sub if a['is_best']) / len(sub) * 100, 1)
                    cumul['avg_loss']       = round(float(np.mean([a['loss'] for a in sub])), 3)
                    cumul['total_steps']    = len(sub)
                    from collections import Counter as _Counter
                    cumul['player_actions'] = dict(_Counter(a['action_name']      for a in sub))
                    cumul['ai_actions']     = dict(_Counter(a['best_action_name'] for a in sub))

            payload = {
                'frames':               encode_frames(rendered),
                'gradcam_human':        gcam_human_frames,
                'gradcam_agent':        gcam_agent_frames,
                'has_gradcam':          gcam is not None,
                'human_actions':        h_log,
                'agent_actions':        a_log,
                'summary':              summary,
                'cumulative':           cumul,
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