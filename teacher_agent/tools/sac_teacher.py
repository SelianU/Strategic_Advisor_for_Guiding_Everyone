"""
teacher_agent/tools/sac_teacher.py
────────────────────────────────────
CLI 진입점.  실제 로직은 각 모듈에 위임.

사용법:
  python sac_teacher.py --train
  python sac_teacher.py --train --resume teacher_agent/checkpoints/sac_teacher_frame_1000000.pth
  python sac_teacher.py --test  --model  teacher_agent/checkpoints/best_model.pth
  python sac_teacher.py --test  --model  teacher_agent/checkpoints/best_model.pth --render
  python sac_teacher.py --train --env ALE/Breakout-v5
"""
import sys
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from .config  import Config
from .trainer import train, test


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAC (Discrete) Teacher Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train",        action="store_true",  help="학습 모드")
    parser.add_argument("--test",         action="store_true",  help="테스트 모드")
    parser.add_argument("--model",        type=str,  default=None, help="테스트할 모델 경로")
    parser.add_argument("--resume",       type=str,  default=None, help="이어서 학습할 체크포인트 경로")
    parser.add_argument("--render",       action="store_true",  help="게임 화면 렌더링")
    parser.add_argument("--episodes",     type=int,  default=10,   help="테스트 에피소드 수")
    parser.add_argument("--env",          type=str,  default=None, help="환경 ID (기본: ALE/SpaceInvaders-v5)")
    parser.add_argument("--lr",           type=float,default=None, help="Q/Policy 공통 LR")
    parser.add_argument("--total-frames", type=int,  default=None, help="총 학습 프레임 수")
    parser.add_argument("--no-autotune",  action="store_true",  help="엔트로피 자동 튜닝 비활성화")
    parser.add_argument("--alpha",        type=float,default=None, help="고정 alpha (--no-autotune 시 사용)")
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()
    config = Config()

    if args.env:          config.ENV_NAME     = args.env
    if args.lr:           config.POLICY_LR    = config.Q_LR = args.lr
    if args.total_frames: config.TOTAL_FRAMES = args.total_frames
    if args.no_autotune:  config.AUTOTUNE     = False
    if args.alpha:        config.ALPHA        = args.alpha

    if args.train:
        train(config, resume_path=args.resume)
    elif args.test:
        if not args.model:
            parser.error("--test requires --model MODEL_PATH")
        test(args.model, config, n_episodes=args.episodes, render=args.render)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
