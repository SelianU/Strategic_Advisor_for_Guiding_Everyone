"""
AdroitBowlingEnv  –  Adroit 손으로 볼링공을 굴려 핀을 맞추는 RL 환경
=======================================================================

환경 구조
---------
1. throw_arm 바디 (arm_slide 슬라이드 관절)
   - Y축 방향으로 슬라이드 → 볼링 스윙 동작
2. Adroit 손 (손목 + 5개 손가락, DAPG_Adroit.xml)
   - 중지·약지가 볼링공 구멍에 삽입된 그립 자세
3. 볼링공 (free joint)
   - equality weld 로 팜에 고정 → 팔이 충분히 앞으로 나오면 자동 해제
4. 핀 4개 (마름모 대형, 각각 free joint)

에피소드 단계 (Phase)
----------------------
Phase 1 – 그립 & 스윙 (ball_released = False):
    팔이 Y축 방향으로 앞으로 나올수록 보상
Phase 2 – 롤 & 핀 (ball_released = True):
    공의 전방 속도와 쓰러진 핀 수에 비례한 보상

관측 (obs) 벡터, 총 71차원
---------------------------
  [0]      arm_slide qpos         (1)
  [1]      arm_slide qvel         (1)
  [2:26]   손 관절 각도 (24관절)   (24)
  [26:50]  손 관절 속도            (24)
  [50:53]  볼 위치 xyz             (3)
  [53:59]  볼 선속도+각속도 (6)    (6)
  [59:71]  핀 1~4 위치 xyz        (12)

액션 공간: 25차원 연속 (DAPG 손 24개 + arm_slide 1개)
  [0:1]     손목 (A_WRJ1, A_WRJ0)          ← DAPG_assets.xml 정의
  [2:5]     검지 (A_FFJ3~A_FFJ0)
  [6:9]     중지 (A_MFJ3~A_MFJ0)
  [10:13]   약지 (A_RFJ3~A_RFJ0)
  [14:18]   새끼 (A_LFJ4~A_LFJ0)
  [19:23]   엄지 (A_THJ4~A_THJ0)
  [24]      arm_slide (A_arm_slide)         ← bowling.xml 추가
"""

import os
import numpy as np
from gym import utils
from mjrl.envs import mujoco_env


# ── 상수 ────────────────────────────────────────────────────────────────
PIN_NAMES        = ['pin_1', 'pin_2', 'pin_3', 'pin_4']
NUM_PINS         = len(PIN_NAMES)

# throw_arm 의 XML pos="0 -0.2 -0.35", wrist +0.396 z, palm +0.034 z
# S_grasp at palm +(0.007, -0.05, 0.07) → 월드 약 (0.007, -0.25, 0.150)
BALL_INIT_POS    = np.array([0.007, -0.25, 0.150])
BALL_RADIUS      = 0.11

# 핀의 직립 기준 z 좌표 (body z = 0.19)
PIN_UPRIGHT_Z    = 0.19
# 이 z 이하면 넘어진 것으로 판정
PIN_TOPPLE_Z     = 0.10

# arm_slide 가 이 값을 넘으면 weld 해제 → 공이 자유낙하 후 굴러감
RELEASE_THRESHOLD = 0.35

# 핀의 초기 위치 [x, y, z, qw, qx, qy, qz]
PIN_INIT_POSES = [
    np.array([ 0.00, 2.80, PIN_UPRIGHT_Z,  1, 0, 0, 0]),
    np.array([-0.16, 3.00, PIN_UPRIGHT_Z,  1, 0, 0, 0]),
    np.array([ 0.16, 3.00, PIN_UPRIGHT_Z,  1, 0, 0, 0]),
    np.array([ 0.00, 3.20, PIN_UPRIGHT_Z,  1, 0, 0, 0]),
]

# 초기 손가락 굽힘 각도 (라디안) – 볼링 그립 자세
# 중지·약지는 공 구멍에 맞게 크게 굽힘, 엄지는 옆에서 감쌈
FINGER_GRIP = {
    'MFJ2': 1.0, 'MFJ1': 0.9, 'MFJ0': 0.7,   # 중지 (볼 구멍)
    'RFJ2': 1.0, 'RFJ1': 0.9, 'RFJ0': 0.7,   # 약지 (볼 구멍)
    'THJ4': 0.35,'THJ3': 0.85,'THJ2': 0.10,  # 엄지 (볼 옆면)
    'FFJ2': 0.5, 'FFJ1': 0.4,                 # 검지 (약간 굽힘)
    'LFJ2': 0.4, 'LFJ1': 0.3,                 # 새끼 (약간 굽힘)
}

# ── 환경 클래스 ──────────────────────────────────────────────────────────
class AdroitBowlingEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(curr_dir, 'bowling.xml')
        # MujocoEnv.__init__ 내부에서 step()을 먼저 호출하므로
        # 플래그와 상태를 미리 초기화해 두어야 함
        self._ids_initialized = False
        self._ball_released   = False
        self._pins_toppled    = [False] * NUM_PINS
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip=4)
        # 혹시 lazy-init 이 아직 안 됐으면 여기서 보장
        if not self._ids_initialized:
            self._setup_ids()

    # ── ID 캐싱 ───────────────────────────────────────────────────────
    def _setup_ids(self):
        """모델 로드 후 자주 쓰는 인덱스를 캐싱."""
        m = self.model

        # 볼링공 바디
        self.ball_body_id   = m.body_name2id('bowling_ball')
        ball_jid            = m.joint_name2id('ball_freejoint')
        self.ball_qpos_adr  = m.jnt_qposadr[ball_jid]  # 7 (xyz + quat)
        self.ball_qvel_adr  = m.jnt_dofadr[ball_jid]   # 6 (v + omega)

        # 팔 슬라이드 관절
        arm_jid             = m.joint_name2id('arm_slide')
        self.arm_qpos_adr   = m.jnt_qposadr[arm_jid]
        self.arm_qvel_adr   = m.jnt_dofadr[arm_jid]

        # 핀 바디 ID & free-joint qpos 주소
        self.pin_body_ids   = [m.body_name2id(n) for n in PIN_NAMES]
        self.pin_qpos_adrs  = []
        for i in range(1, NUM_PINS + 1):
            jid = m.joint_name2id(f'pin{i}_joint')
            self.pin_qpos_adrs.append(m.jnt_qposadr[jid])

        # 손 관절 이름 목록 (obs 추출 순서)
        self.hand_joint_names = [
            'WRJ1', 'WRJ0',
            'FFJ3', 'FFJ2', 'FFJ1', 'FFJ0',
            'MFJ3', 'MFJ2', 'MFJ1', 'MFJ0',
            'RFJ3', 'RFJ2', 'RFJ1', 'RFJ0',
            'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'LFJ0',
            'THJ4', 'THJ3', 'THJ2', 'THJ1', 'THJ0',
        ]
        self.hand_qpos_adrs = []
        self.hand_qvel_adrs = []
        for jname in self.hand_joint_names:
            jid = m.joint_name2id(jname)
            self.hand_qpos_adrs.append(m.jnt_qposadr[jid])
            self.hand_qvel_adrs.append(m.jnt_dofadr[jid])

        # ball_grip equality constraint ID 탐색
        self.ball_grip_eq_id = self._find_eq_id('ball_grip')

        # 에피소드 상태 플래그
        self._ball_released = False
        self._pins_toppled  = [False] * NUM_PINS
        self._ids_initialized = True  # 초기화 완료 표시

    def _find_eq_id(self, name: str) -> int:
        """equality constraint 이름으로 인덱스 반환."""
        for i in range(self.model.neq):
            try:
                eq_name = self.model.eq_id2name(i)
            except Exception:
                # 일부 mujoco-py 버전에서 eq_id2name 없으면 index 0 사용
                return 0
            if eq_name == name:
                return i
        # 못 찾으면 0 (ball_grip 이 유일한 equality 이므로)
        return 0

    # ── weld 제어 ─────────────────────────────────────────────────────
    def _set_grip(self, active: bool):
        """ball_grip equality weld 를 활성화/비활성화."""
        self.model.eq_active[self.ball_grip_eq_id] = int(active)

    # ── step ──────────────────────────────────────────────────────────
    def step(self, action):
        # MujocoEnv.__init__가 step()을 먼저 호출할 수 있으므로 lazy 초기화
        if not self._ids_initialized:
            self._setup_ids()

        # 1. 공 해제 조건 확인 (시뮬레이션 전)
        arm_pos = float(self.data.qpos[self.arm_qpos_adr])
        if not self._ball_released and arm_pos >= RELEASE_THRESHOLD:
            self._ball_released = True
            self._set_grip(False)  # weld 해제 → 공이 물리 법칙에 따라 굴러감

        # 2. 시뮬레이션
        self.do_simulation(action, self.frame_skip)

        # 3. 상태 수집
        ball_pos = self.data.body_xpos[self.ball_body_id].copy()
        ball_vel = self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 6].copy()
        arm_vel  = float(self.data.qvel[self.arm_qvel_adr])
        arm_pos  = float(self.data.qpos[self.arm_qpos_adr])  # 업데이트

        # 핀 넘어짐 판정
        for i, bid in enumerate(self.pin_body_ids):
            if not self._pins_toppled[i]:
                pin_z = self.data.body_xpos[bid][2]
                if pin_z < PIN_TOPPLE_Z:
                    self._pins_toppled[i] = True

        pins_down = int(sum(self._pins_toppled))

        # ── 4. 보상 계산 ──────────────────────────────────────────────
        reward = 0.0

        if not self._ball_released:
            # Phase 1 – 스윙 단계
            # 팔이 앞으로 나올수록, 빠를수록 보상
            reward += 2.0 * arm_pos              # 전진 거리 보상
            reward += 1.5 * max(arm_vel, 0.0)   # 전방 속도 보상
            # 손목이 적절히 구부러져 공을 레인 방향으로 향하도록 유도
            wrist_pitch = float(self.data.qpos[
                self.model.jnt_qposadr[self.model.joint_name2id('WRJ1')]
            ])
            reward += 0.5 * max(wrist_pitch, 0.0)  # 손목 앞으로 꺾기

        else:
            # Phase 2 – 롤링 & 핀 단계
            forward_vel = ball_vel[1]            # Y축 속도 (전방)
            lateral_dev = abs(ball_pos[0])       # X축 이탈 (레인 중앙에서)

            reward += 3.0 * max(forward_vel, 0.0)   # 전방 구름 속도
            reward -= 1.5 * lateral_dev              # 가터 진입 페널티
            reward += 15.0 * pins_down               # 넘어진 핀 수 (즉시 반영)

            # 공이 핀 쪽으로 잘 향하고 있으면 보너스
            if ball_pos[1] > 1.5 and forward_vel > 0.5:
                reward += 1.0

        # ── 5. 종료 조건 ──────────────────────────────────────────────
        done = False

        # 공이 레인을 벗어났거나 뒤로 나간 경우
        if ball_pos[1] > 4.5:          # 레인 끝 통과 (핀 타격 후)
            done = True
        elif abs(ball_pos[0]) > 1.65:  # 가터 이탈
            done = True
        elif ball_pos[2] < -0.5:       # 낙하 (이상 상태)
            done = True

        obs = self._get_obs()
        info = {
            'pins_toppled': pins_down,
            'ball_released': self._ball_released,
            'arm_slide':  arm_pos,
            'ball_pos':   ball_pos.tolist(),
        }
        return obs, float(reward), done, info

    # ── 관측 ─────────────────────────────────────────────────────────
    def _get_obs(self):
        """
        71차원 관측 벡터 반환.
        (상단 docstring 참조)
        """
        # 팔 슬라이드
        arm_q = np.array([self.data.qpos[self.arm_qpos_adr]])
        arm_v = np.array([self.data.qvel[self.arm_qvel_adr]])

        # 손 관절 (24개 관절)
        hand_qpos = np.array([self.data.qpos[a] for a in self.hand_qpos_adrs])
        hand_qvel = np.array([self.data.qvel[a] for a in self.hand_qvel_adrs])

        # 볼링공
        ball_pos = self.data.body_xpos[self.ball_body_id].copy()
        ball_vel = self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 6].copy()

        # 핀 위치
        pin_positions = np.concatenate(
            [self.data.body_xpos[bid].copy() for bid in self.pin_body_ids]
        )  # (12,)

        return np.concatenate([
            arm_q, arm_v,           # 2
            hand_qpos, hand_qvel,   # 48
            ball_pos, ball_vel,     # 9
            pin_positions,          # 12
        ])  # 총 71

    # ── 리셋 ─────────────────────────────────────────────────────────
    def reset_model(self):
        """
        에피소드 시작 시 호출:
        - 팔을 뒤로 당김 (arm_slide = 0)
        - 손가락을 볼링 그립 자세로 설정
        - 공을 S_grasp 위치에 배치 + weld 재활성화
        - 핀을 직립 위치로 복원
        """
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # ── 핀 위치 초기화 ────────────────────────────────────────────
        for padr, pose in zip(self.pin_qpos_adrs, PIN_INIT_POSES):
            qpos[padr : padr + 7] = pose

        # ── 볼링공 초기 위치 (S_grasp 근처) ──────────────────────────
        # arm_slide = 0 일 때 월드 좌표 ≈ (0.007, -0.25, 0.150)
        qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = BALL_INIT_POS
        qpos[self.ball_qpos_adr + 3] = 1.0  # quaternion w
        qpos[self.ball_qpos_adr + 4] = 0.0
        qpos[self.ball_qpos_adr + 5] = 0.0
        qpos[self.ball_qpos_adr + 6] = 0.0

        # ── 팔 뒤로 당김 ─────────────────────────────────────────────
        qpos[self.arm_qpos_adr] = 0.0

        # ── 손가락 볼링 그립 자세 ────────────────────────────────────
        for jname, angle in FINGER_GRIP.items():
            try:
                jid  = self.model.joint_name2id(jname)
                qadr = self.model.jnt_qposadr[jid]
                qpos[qadr] = angle
            except Exception:
                pass  # 해당 관절 없으면 스킵

        self.set_state(qpos, qvel)

        # ── weld 재활성화 + 플래그 초기화 ────────────────────────────
        self._set_grip(True)
        self._ball_released = False
        self._pins_toppled  = [False] * NUM_PINS

        return self._get_obs()

    def viewer_setup(self):
        """뷰어 기본 시점을 볼링공 추적으로 설정."""
        self.viewer.cam.trackbodyid = self.ball_body_id
        self.viewer.cam.distance    = 3.5
        self.viewer.cam.elevation   = -18
        self.viewer.cam.azimuth     = 0


# ── 빠른 동작 테스트 ─────────────────────────────────────────────────────
if __name__ == '__main__':
    import imageio

    print("=" * 55)
    print(" 볼링 RL 환경 생성 중...")
    print("=" * 55)
    env = AdroitBowlingEnv()

    obs = env.reset()
    print(f"✅ 환경 생성 성공!")
    print(f"   obs  shape : {obs.shape}")          # (71,)
    print(f"   action dim : {env.action_space.shape[0]}")  # 25

    print("\n무작위 행동으로 300 스텝 테스트 중...")
    frames = []
    total_pins = 0

    for step in range(300):
        frame = env.sim.render(
            width=640, height=480, camera_name='bowl_back'
        )[::-1, :, :]
        frames.append(frame)

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_pins = max(total_pins, info['pins_toppled'])

        if step % 60 == 0:
            print(f"  step={step:3d} | arm={info['arm_slide']:.3f} | "
                  f"released={info['ball_released']} | "
                  f"pins_down={info['pins_toppled']}/{NUM_PINS} | "
                  f"reward={reward:+.2f}")
        if done:
            obs = env.reset()

    env.close()

    print(f"\n테스트 완료. 최대 쓰러진 핀: {total_pins}/{NUM_PINS}")
    print("동영상 저장 중...")
    imageio.mimsave('bowling_test.mp4', frames, fps=30)
    print("✅ 'bowling_test.mp4' 저장 완료!")