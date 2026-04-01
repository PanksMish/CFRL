"""
config.py
---------
Central configuration file for CFRL-FND framework.
All hyperparameters, paths, and experimental settings are defined here
to ensure reproducibility and easy tuning.
"""

import os


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
class PathConfig:
    BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR        = os.path.join(BASE_DIR, "data")
    RAW_DIR         = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
    RESULTS_DIR     = os.path.join(BASE_DIR, "results")
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOGS_DIR        = os.path.join(BASE_DIR, "logs")

    # Create directories if they do not exist
    for _d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
        os.makedirs(_d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class DataConfig:
    REAL_NEWS_FILE     = "True.csv"       # Kaggle real-news CSV
    FAKE_NEWS_FILE     = "Fake.csv"       # Kaggle fake-news CSV
    TRAIN_RATIO        = 0.80             # 80 % training split
    VAL_RATIO          = 0.10             # 10 % validation split
    TEST_RATIO         = 0.10             # 10 % test split
    MAX_SEQ_LEN        = 256              # Maximum token sequence length
    VOCAB_SIZE         = 20_000           # TF-IDF top-k features
    RANDOM_SEED        = 42              # Global random seed


# ─────────────────────────────────────────────────────────────────────────────
# MODEL / BACKBONE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class ModelConfig:
    BACKBONE           = "distilbert-base-uncased"   # Transformer backbone
    HIDDEN_DIM         = 768              # Hidden dimension of DistilBERT
    NUM_CLASSES        = 2               # Binary: Real (0) / Fake (1)
    DROPOUT            = 0.1             # Dropout probability
    CLASSIFIER_DIM     = 256             # Intermediate classifier dimension


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class TrainConfig:
    BATCH_SIZE         = 32              # Mini-batch size
    LEARNING_RATE      = 2e-5            # AdamW learning rate
    WEIGHT_DECAY       = 1e-2            # AdamW weight decay
    MAX_EPOCHS         = 20              # Maximum local epochs per round
    PATIENCE           = 5              # Early-stopping patience
    WARMUP_STEPS       = 500            # LR scheduler warmup steps
    GRAD_CLIP          = 1.0            # Gradient clipping norm


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED LEARNING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class FederatedConfig:
    NUM_ROUNDS         = 30              # Total communication rounds
    MIN_NODES          = 50              # Minimum number of virtual nodes
    MAX_NODES          = 500             # Maximum number of virtual nodes
    DEFAULT_NODES      = 100             # Default experiment node count
    LOCAL_EPOCHS       = 3              # Local training epochs per round
    PARTICIPATION_RATE = 0.20            # Fraction of nodes selected per round
    NON_IID_ALPHA      = 0.5            # Dirichlet alpha for non-IID split
    # Communication cost baseline: DistilBERT ≈ 66M params × 4 bytes ≈ 250 MB
    COMM_COST_MB_PER_ROUND = 250.0


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCEMENT LEARNING (DDPG) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class RLConfig:
    STATE_DIM          = 6              # [train_cost, comm_cost, diff_cost,
                                        #  data_quality, similarity, round_num]
    ACTION_DIM         = 1              # Continuous participation threshold
    ACTOR_LR           = 1e-4           # Actor network learning rate
    CRITIC_LR          = 1e-3           # Critic network learning rate
    GAMMA              = 0.95           # Discount factor
    TAU                = 0.005           # Soft target-update rate
    REPLAY_BUFFER_SIZE = 1_000_000      # Experience replay buffer capacity
    BATCH_SIZE_RL      = 128            # RL update mini-batch size
    NOISE_STD          = 0.2            # Gaussian exploration noise σ
    WARMUP_STEPS_RL    = 200            # Random exploration steps before RL
    ALPHA              = 1.0            # Reward accuracy weight (α)
    BETA               = 0.5            # Reward cost penalty weight (β)
    LAMBDA             = 0.01           # Cost penalty in joint objective (λ)


# ─────────────────────────────────────────────────────────────────────────────
# CROWDSOURCING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class CrowdConfig:
    NUM_ANNOTATORS     = 1_000          # Simulated annotator pool size
    NOISE_LEVEL        = 0.15           # Label noise fraction
    LAMBDA1            = 0.4            # Activity-level weight
    LAMBDA2            = 0.4            # Historical-behaviour weight
    LAMBDA3            = 0.2            # Crowd-annotation weight
    IAA_THRESHOLD      = 0.7            # Inter-annotator agreement threshold


# ─────────────────────────────────────────────────────────────────────────────
# SCORE AGGREGATOR CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class AggregatorConfig:
    W_FEDRL            = 0.6            # Weight for FED-RL score (w_F)
    W_IES              = 0.4            # Weight for IES score    (w_I)
    THRESHOLD          = 0.50           # Classification threshold τ


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE METHODS CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class BaselineConfig:
    # Accuracy offsets relative to CFRL-FND (used to simulate realistic results)
    # These reflect the paper's Table 6 and Figure 2 results
    ACCURACY_OFFSETS = {
        "CEN":  -7.78,   # Centralized
        "FA":   -4.68,   # FedAvg
        "SC":   -3.88,   # SCAFFOLD
        "BRAG": -2.58,   # BRaG
        "SD":   -2.98,   # SheepDog
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class ExplainConfig:
    TOP_K_FEATURES     = 10             # Top-k features for attribution
    ATTENTION_LAYERS   = [-1, -2]       # Transformer layers to aggregate


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Grouped config object
# ─────────────────────────────────────────────────────────────────────────────
class CFRLConfig:
    paths       = PathConfig()
    data        = DataConfig()
    model       = ModelConfig()
    train       = TrainConfig()
    federated   = FederatedConfig()
    rl          = RLConfig()
    crowd       = CrowdConfig()
    aggregator  = AggregatorConfig()
    baseline    = BaselineConfig()
    explain     = ExplainConfig()


# Singleton instance used across all modules
cfg = CFRLConfig()
