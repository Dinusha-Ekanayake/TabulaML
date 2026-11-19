"""Central constants and thresholds. No dataset-specific values belong here."""

RANDOM_STATE = 42

# Tokens treated as missing (compared lowercase after stripping).
MISSING_TOKENS: tuple[str, ...] = (
    "",
    "?",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "-",
    "--",
    "missing",
)
TRUE_TOKENS: tuple[str, ...] = ("yes", "y", "true", "t")
FALSE_TOKENS: tuple[str, ...] = ("no", "n", "false", "f")

# Input limits.
MAX_FILE_MB = 200
MAX_CELLS = 1_000_000
MIN_ROWS = 30

# Profiling / recommendation thresholds.
MOSTLY_MISSING = 0.60
ID_UNIQUE_RATIO = 0.95
HIGH_CARD_THRESHOLD = 20
RARE_MIN_FREQ = 0.01
SKEW_THRESHOLD = 1.0
OUTLIER_RATIO_CLIP = 0.01
OUTLIER_RATIO_ROBUST = 0.05
IMBALANCE_MINORITY = 0.20
TEXT_AVG_LEN = 30

# Leakage detection.
LEAKAGE_CORR = 0.95
LEAKAGE_AUC = 0.98
