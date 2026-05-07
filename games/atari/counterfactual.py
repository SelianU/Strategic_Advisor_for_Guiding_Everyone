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
                    h_stack.append(preprocess(obs_h))
                    h_score += float(rew_h)
                    if rew_h > 0:
                        h_steps.append(offset + 1)
                        if h_first is None:
                            h_first = offset + 1
                    h_done = t_h or tr_h

                if not a_done:
                    obs_a, rew_a, t_a, tr_a, _ = agent_env.step(a_action)
                    a_frame = obs_a
                    a_stack.append(preprocess(obs_a))
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