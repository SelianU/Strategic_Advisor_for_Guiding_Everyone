import os
import csv
import time

import ale_py
import gymnasium as gym
import pygame
from PIL import Image
import numpy as np


# =========================
# 1. 저장 경로 설정
# =========================
BASE_DIR = "human_logs"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CSV_PATH = os.path.join(BASE_DIR, "actions.csv")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)


# =========================
# 2. CSV 초기화
# =========================
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "step_in_episode",
            "global_step",
            "action",
            "reward",
            "terminated",
            "truncated",
            "image_path"
        ])


# =========================
# 3. 환경 생성
# =========================
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
obs, info = env.reset(seed=42)

print("observation shape:", obs.shape)
print("action space:", env.action_space)

# Pong action 참고:
# 0 = NOOP
# 1 = FIRE
# 2 = UP
# 3 = DOWN
# 4 = UPFIRE
# 5 = DOWNFIRE
#
# 여기서는 단순화를 위해:
#   위 화살표 -> 2 (UP)
#   아래 화살표 -> 3 (DOWN)
#   아무 키 없음 -> 0 (NOOP)
#   스페이스바 -> 1 (FIRE)

# =========================
# 4. pygame 초기화
# =========================
pygame.init()

frame_height, frame_width, _ = obs.shape
screen = pygame.display.set_mode((frame_width, frame_height))
pygame.display.set_caption("Human Pong Player")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

running = True
episode = 1
step_in_episode = 0
global_step = 0

print("\n조작법:")
print("  ↑ : 위로 이동")
print("  ↓ : 아래로 이동")
print("  SPACE : FIRE")
print("  아무 키도 안 누르면 NOOP")
print("  ESC 또는 창 닫기 : 종료\n")


def get_human_action() -> int:
    """키 입력을 Pong action으로 변환"""
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        return 2  # UP
    elif keys[pygame.K_DOWN]:
        return 3  # DOWN
    elif keys[pygame.K_SPACE]:
        return 1  # FIRE
    else:
        return 0  # NOOP


def render_frame(frame: np.ndarray, current_action: int, reward: float, episode_num: int, ep_step: int):
    """Atari frame을 pygame 창에 띄우고 간단한 텍스트 표시"""
    # Gymnasium frame은 (H, W, C), pygame은 (W, H, C)처럼 다뤄야 해서 transpose
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    screen.blit(surface, (0, 0))

    info_text = f"Episode: {episode_num} | Step: {ep_step} | Action: {current_action} | Reward: {reward}"
    text_surface = font.render(info_text, True, (255, 255, 255))
    screen.blit(text_surface, (5, 5))

    pygame.display.flip()


def save_step_data(frame: np.ndarray, episode_num: int, ep_step: int, g_step: int,
                   action: int, reward: float, terminated: bool, truncated: bool):
    """현재 프레임 이미지와 메타데이터 저장"""
    image_filename = f"ep{episode_num:03d}_step{ep_step:04d}_global{g_step:05d}.png"
    image_path = os.path.join(IMAGE_DIR, image_filename)

    Image.fromarray(frame).save(image_path)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode_num,
            ep_step,
            g_step,
            action,
            reward,
            int(terminated),
            int(truncated),
            image_path
        ])


# =========================
# 5. 메인 루프
# =========================
last_reward = 0.0

while running:
    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    action = get_human_action()

    next_obs, reward, terminated, truncated, info = env.step(action)

    step_in_episode += 1
    global_step += 1
    last_reward = reward

    # 저장
    save_step_data(
        frame=obs,
        episode_num=episode,
        ep_step=step_in_episode,
        g_step=global_step,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated
    )

    # 화면 표시
    render_frame(obs, action, reward, episode, step_in_episode)

    # 디버그 출력
    print(
        f"[episode {episode}] [step {step_in_episode}] [global {global_step}] "
        f"action={action}, reward={reward}, terminated={terminated}, truncated={truncated}"
    )

    obs = next_obs

    if terminated or truncated:
        print(f"=== episode {episode} 종료 (길이: {step_in_episode}) ===")
        obs, info = env.reset()
        episode += 1
        step_in_episode = 0

    # 너무 빠르면 플레이가 어려우니 FPS 제한
    clock.tick(30)

env.close()
pygame.quit()
print("종료되었습니다.")
