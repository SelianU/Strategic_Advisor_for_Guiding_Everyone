"""
record_play.py  —  D3QN_v3 best_model.pth 플레이 영상 녹화 스크립트

사용법:
    python record_play.py
    python record_play.py --model ./checkpoints_v3/best_model.pth
    python record_play.py --model ./checkpoints_v3/best_model.pth --episodes 3 --out ./videos

출력:
    ./videos/episode_1.mp4
    ./videos/episode_2.mp4
    ...
    ./videos/best_episode.mp4   ← 가장 높은 점수 에피소드 별도 저장
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch

import gymnasium as gym
import ale_py  # noqa: F401

# D3QN_v3.py와 같은 디렉토리에 있어야 합니다.
# 없으면 아래 경로를 D3QN_v3.py 실제 경로로 수정하세요.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from D3QN_v3 import Config, DuelingDQN, AtariWrapper, make_env


# =============================================================================
# 설정
# =============================================================================
DEFAULT_MODEL   = "./checkpoints_v3/best_model.pth"
DEFAULT_OUT_DIR = "./videos"
DEFAULT_EPISODES = 3
EVAL_EPSILON     = 0.001   # 거의 greedy (1034점 달성한 eval 세팅과 동일)
FPS              = 30      # 영상 FPS


# =============================================================================
# 모델 로드 (D3QNAgent 전체를 import하지 않고 네트워크만 로드)
# =============================================================================
def load_model(model_path: str, device: torch.device):
    config = Config()

    # 환경에서 input_shape, n_actions 가져오기
    env_tmp = make_env(config)
    obs, _ = env_tmp.reset()
    input_shape = obs.shape          # (4, 84, 84)
    n_actions   = env_tmp.action_space.n
    env_tmp.close()

    net = DuelingDQN(input_shape, n_actions).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    # q_network 키가 있으면 바로 로드, 없으면 state_dict 자체로 시도
    state_dict = ckpt.get('q_network', ckpt)

    # torch.compile로 저장된 경우 '_orig_mod.' 접두사 제거
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace('_orig_mod.', '')] = v

    net.load_state_dict(cleaned)
    net.eval()

    frame_num = ckpt.get('frame', 0)
    print(f"  모델 로드 완료: {model_path}")
    print(f"  학습 완료 frame: {frame_num:,}")
    print(f"  input_shape={input_shape}, n_actions={n_actions}")
    return net, input_shape, n_actions


# =============================================================================
# 단일 에피소드 녹화
# =============================================================================
def record_episode(net, config, device, out_path: str, ep_idx: int):
    """
    rgb_array 모드로 환경을 실행하며 각 프레임을 MP4로 저장.
    반환값: 에피소드 총 점수
    """
    # rgb_array 모드: gym이 render() 호출 시 ndarray 반환
    # [BUG FIX] frameskip=1 필수:
    #   AtariWrapper.step()이 내부에서 FRAME_SKIP=3번 반복하므로
    #   여기서 frameskip=3으로 설정하면 3x3=9가 되어 eval과 환경이 달라짐
    env = gym.make(
        config.ENV_NAME,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="rgb_array",
    )
    env = AtariWrapper(env, config)

    obs, _ = env.reset()
    state  = obs   # AtariWrapper가 이미 (4,84,84) uint8 반환

    # VideoWriter 초기화 (첫 프레임 크기로 설정)
    raw_frame = env.render()
    h, w = raw_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (w, h))

    score = 0.0
    step  = 0
    done  = False

    while not done:
        # 프레임 저장
        raw_frame = env.render()
        bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

        # 점수/스텝 오버레이
        cv2.putText(bgr, f"Score: {int(score)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(bgr, f"Step:  {step}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        writer.write(bgr)

        # 행동 선택 (greedy)
        with torch.no_grad():
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
            # DuelingDQN.forward()는 내부에서 /255 하므로 uint8 * 1.0 전달
            s_t_raw = torch.from_numpy(state.astype(np.uint8)).unsqueeze(0).to(device).float()
            q = net(s_t_raw)

        if np.random.random() < EVAL_EPSILON:
            action = env.action_space.sample()
        else:
            action = q.argmax(dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done   = terminated or truncated
        state  = obs
        score += reward
        step  += 1

    # 마지막 프레임에 최종 점수 표시 (1초간)
    cv2.putText(bgr, f"FINAL: {int(score)}", (w//2 - 100, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    for _ in range(FPS):
        writer.write(bgr)

    writer.release()
    env.close()

    print(f"  Episode {ep_idx+1}: score={int(score):>5}  →  {out_path}")
    return score


# =============================================================================
# 메인
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="D3QN best_model 플레이 녹화")
    parser.add_argument("--model",    default=DEFAULT_MODEL,   help="모델 경로")
    parser.add_argument("--episodes", default=DEFAULT_EPISODES, type=int, help="녹화 에피소드 수")
    parser.add_argument("--out",      default=DEFAULT_OUT_DIR,  help="영상 저장 폴더")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net, _, _ = load_model(args.model, device)
    config    = Config()
    config.REWARD_CLIP           = False
    config.TERMINAL_ON_LIFE_LOSS = False

    os.makedirs(args.out, exist_ok=True)
    scores = []

    for i in range(args.episodes):
        out_path = os.path.join(args.out, f"episode_{i+1}.mp4")
        s = record_episode(net, config, device, out_path, i)
        scores.append((s, out_path))

    # 최고 점수 에피소드 복사
    best_score, best_path = max(scores, key=lambda x: x[0])
    best_out = os.path.join(args.out, "best_episode.mp4")
    import shutil
    shutil.copy(best_path, best_out)

    print()
    print("=" * 50)
    print(f"  녹화 완료: {args.episodes}개 에피소드")
    print(f"  점수:  {[int(s) for s, _ in scores]}")
    print(f"  평균:  {np.mean([s for s, _ in scores]):.1f}")
    print(f"  최고:  {int(best_score)}  →  {best_out}")
    print("=" * 50)


if __name__ == "__main__":
    main()