class Config:
    """
    D3QN + PER 하이퍼파라미터.

    Base papers:
      [1] Mnih et al.       (2013) - DQN
      [2] van Hasselt et al.(2015) - Double DQN
      [3] Wang et al.       (2016) - Dueling Network
      [4] Schaul et al.     (2015) - Prioritized Experience Replay
    """

    # Environment
    ENV_NAME   = "ALE/SpaceInvaders-v5"
    FRAME_SKIP = 15
    STACK_SIZE = 4
    IMG_SIZE   = 84
    NO_OP_MAX  = 30

    # Replay Buffer
    REPLAY_CAPACITY = 1_000_000
    LEARNING_START  = 50_000

    # PER [4]
    PER_ALPHA       = 0.6
    PER_BETA_START  = 0.4
    PER_BETA_FRAMES = 100_000_000
    PER_EPS         = 1e-6

    # Training
    BATCH_SIZE      = 32
    DISCOUNT_FACTOR = 0.99
    LEARNING_RATE   = 6.25e-5
    ADAM_EPS        = 1.5e-4
    TRAIN_FREQ      = 4
    TOTAL_FRAMES    = 100_000_000

    # Target Network
    SOFT_UPDATE        = False
    TAU                = 0.005
    TARGET_UPDATE_FREQ = 10_000

    # Exploration
    EPSILON_START = 1.0
    EPSILON_END   = 0.01
    EPSILON_DECAY = 1_000_000

    # Gradient & Reward
    GRAD_CLIP_NORM = 10.0
    REWARD_CLIP    = True
    TERMINAL_ON_LIFE_LOSS = True

    # Evaluation
    EVAL_FREQ     = 250_000
    EVAL_EPISODES = 5
    EVAL_EPSILON  = 0.001

    # Checkpointing & Logging
    CHECKPOINT_FREQ      = 250_000
    MILESTONE_FREQ       = 50_000_000   # milestone 저장 간격
    MAX_CHECKPOINTS      = 10           # 일반 체크포인트 최대 보관 개수
    CHECKPOINT_DIR       = "./checkpoints"        # 최근 N개 rolling 저장
    BEST_CHECKPOINT_DIR  = "./best_checkpoints"   # best_model, milestone, final 저장
    LOG_FREQ             = 10
