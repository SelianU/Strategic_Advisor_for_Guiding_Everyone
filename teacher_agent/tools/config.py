"""
teacher_agent/tools/config.py
──────────────────────────────
SAC (Continuous) Teacher Agent 하이퍼파라미터.

State  : Atari obs (4×84×84) + remaining lives (scalar) concat
Action : 단일 연속값 ∈ [0, 1]  (개입 확률)
Intervene: prob >= THRESHOLD → 개입
"""


class Config:

    # ── Environment ───────────────────────────────────────────────────────────
    ENV_NAME   = "ALE/SpaceInvaders-v5"
    FRAME_SKIP = 4
    STACK_SIZE = 4
    IMG_SIZE   = 84
    NO_OP_MAX  = 30
    MAX_LIVES  = 5          # lives 정규화 분모 (게임마다 다름)

    # ── State / Action ────────────────────────────────────────────────────────
    ACTION_DIM = 1          # 개입 확률 (스칼라)
    THRESHOLD  = 0.65       # prob >= THRESHOLD → 개입

    # ── Replay Buffer ─────────────────────────────────────────────────────────
    REPLAY_CAPACITY = 300_000
    LEARNING_START  = 20_000

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE      = 64
    DISCOUNT_FACTOR = 0.99
    POLICY_LR       = 3e-4
    Q_LR            = 3e-4
    ADAM_EPS        = 1e-4
    UPDATE_FREQ     = 4
    TOTAL_FRAMES    = 5_000_000

    # ── Target Network ────────────────────────────────────────────────────────
    TAU                = 1.0        # Hard update
    TARGET_UPDATE_FREQ = 8_000

    # ── Entropy (Auto-tuning) ─────────────────────────────────────────────────
    ALPHA                = 0.2
    AUTOTUNE             = True
    TARGET_ENTROPY       = -ACTION_DIM   # continuous SAC: -dim(A)

    # ── Reward / Terminal ─────────────────────────────────────────────────────
    REWARD_CLIP           = True
    TERMINAL_ON_LIFE_LOSS = True

    # ── Evaluation ────────────────────────────────────────────────────────────
    EVAL_FREQ     = 250_000
    EVAL_EPISODES = 5

    # ── Checkpointing & Logging ───────────────────────────────────────────────
    CHECKPOINT_FREQ = 250_000
    CHECKPOINT_DIR  = "teacher_agent/checkpoints"
    LOG_FREQ        = 10
