"""
human_play_recorder.py  —  Space Invaders 사람 플레이 기록기

pygame 창에서 직접 키보드로 게임을 플레이하고,
매 스텝의 (raw_frame, state, action, reward, score) 를 pkl 파일로 저장합니다.
저장된 파일은 feedback_analyzer.py 로 분석합니다.

조작:
    ← →        이동
    Space / Z  발사
    E          오른쪽+발사
    Q          왼쪽+발사
    ESC        에피소드 종료 후 저장

사용법:
    python human_play_recorder.py
    python human_play_recorder.py --episodes 2 --out ./human_data/play_001.pkl

출력 포맷 (play_001.pkl):
    List[Episode]
    Episode = {
        "episode_idx" : int,
        "total_score" : float,
        "steps"       : int,
        "frames"      : List[Frame],
    }
    Frame = {
        "step"      : int,
        "raw_frame" : np.ndarray (210,160,3) uint8   <- 원본 RGB
        "state"     : np.ndarray (4,84,84)  uint8   <- 전처리 4-stack
        "action"    : int  (0~5)
        "reward"    : float
        "score"     : float
        "done"      : bool
    }
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pygame

import gymnasium as gym
import ale_py  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D3QN_v3 import Config, AtariWrapper

# =============================================================================
# 상수
# =============================================================================
ACTION_NAMES = {
    0: "NOOP", 1: "FIRE", 2: "RIGHT",
    3: "LEFT", 4: "RIGHT+FIRE", 5: "LEFT+FIRE",
}

DISPLAY_W, DISPLAY_H = 480, 560
GAME_W,    GAME_H    = 480, 480
FPS_CAP              = 30


def get_action(keys) -> int:
    """pygame 키 상태 -> ALE 행동 번호"""
    left  = keys[pygame.K_LEFT]  or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire  = keys[pygame.K_SPACE] or keys[pygame.K_z]

    if left  and fire: return 5
    if right and fire: return 4
    if left:           return 3
    if right:          return 2
    if fire:           return 1
    return 0


# =============================================================================
# 녹화 루프
# =============================================================================
def record_episodes(n_episodes: int, out_path: str):
    config = Config()
    config.REWARD_CLIP           = False
    config.TERMINAL_ON_LIFE_LOSS = False
    config.NO_OP_MAX             = 1

    env_raw = gym.make(
        config.ENV_NAME,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="rgb_array",
    )
    env = AtariWrapper(env_raw, config)

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("Space Invaders — Human Play Recorder")
    clock  = pygame.time.Clock()
    font_s = pygame.font.SysFont("monospace", 15)

    all_episodes = []

    print("=" * 52)
    print("  Space Invaders 플레이 기록기 시작")
    print("=" * 52)
    print("  조작: <- -> 이동   Space/Z 발사")
    print("        E=우+발사   Q=좌+발사   ESC=종료")
    print("=" * 52)

    for ep_idx in range(n_episodes):
        state, _ = env.reset()
        raw_frame = env_raw.render()

        episode_frames = []
        score   = 0.0
        step    = 0
        done    = False
        running = True

        print(f"\n  [Episode {ep_idx+1}/{n_episodes}] 시작!")

        while running and not done:
            clock.tick(FPS_CAP)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("  ESC -> 에피소드 종료")
                        running = False

            if not running:
                break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                action = 5
            elif keys[pygame.K_e]:
                action = 4
            else:
                action = get_action(keys)

            next_state, reward, terminated, truncated, _ = env.step(action)
            raw_frame_new = env_raw.render()
            done   = terminated or truncated
            score += reward
            step  += 1

            episode_frames.append({
                "step":      step,
                "raw_frame": raw_frame.copy(),
                "state":     state.copy(),
                "action":    action,
                "reward":    reward,
                "score":     score,
                "done":      done,
            })

            state     = next_state
            raw_frame = raw_frame_new

            # 화면 렌더링
            surf = pygame.surfarray.make_surface(raw_frame.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (GAME_W, GAME_H))
            screen.fill((10, 10, 10))
            screen.blit(surf, (0, 0))
            pygame.draw.rect(screen, (18, 18, 18), (0, GAME_H, DISPLAY_W, DISPLAY_H - GAME_H))

            hud_lines = [
                (f"Score: {int(score)}", (80, 220, 100)),
                (f"Step: {step}  |  Ep: {ep_idx+1}/{n_episodes}", (160, 160, 160)),
                (f"Action: {ACTION_NAMES[action]}", (100, 180, 255)),
                ("ESC: 종료 후 저장", (100, 100, 100)),
            ]
            for i, (text, color) in enumerate(hud_lines):
                screen.blit(font_s.render(text, True, color), (10, GAME_H + 8 + i * 18))

            pygame.display.flip()

        all_episodes.append({
            "episode_idx": ep_idx,
            "total_score": score,
            "steps":       step,
            "frames":      episode_frames,
        })
        print(f"  Episode {ep_idx+1} 완료: score={int(score)}, steps={step}")

        if not running:
            break

    pygame.quit()
    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_episodes, f)

    total_steps = sum(ep["steps"] for ep in all_episodes)
    print(f"\n  저장 완료 -> {out_path}")
    print(f"  에피소드: {len(all_episodes)}개  |  총 스텝: {total_steps:,}")
    print(f"\n  피드백 분석 명령:")
    print(f"  python feedback_analyzer.py --data {out_path}")


# =============================================================================
# 메인
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",      default="./human_data/play_001.pkl")
    parser.add_argument("--episodes", default=1, type=int)
    args = parser.parse_args()
    record_episodes(args.episodes, args.out)


if __name__ == "__main__":
    main()