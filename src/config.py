"""Central configuration for the NIDS pipeline."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# ─── Dataset Configurations ───────────────────────────────────────────
# Supported: "unsw_nb15", "cicids2018", "cicids2017"
DATASET_NAME = "unsw_nb15"

DATASETS = {
    "unsw_nb15": {
        "label_column": "label",
        "attack_column": "attack_cat",
        "csv_files": [
            "UNSW_NB15_training-set.csv",
            "UNSW_NB15_testing-set.csv",
        ],
        "drop_columns": ["id", "attack_cat"],
        "sample_fraction": 1.0,
        "min_samples_per_class": 500,
        "description": "UNSW-NB15 (2019) — 2.5M flows, 9 attack types, UNSW Canberra",
        "kaggle_slug": "mrwellsdavid/unsw-nb15",
    },
    "cicids2018": {
        "label_column": "Label",
        "attack_column": None,
        "csv_files": [
            "02-14-2018.csv",
            "02-15-2018.csv",
            "02-16-2018.csv",
            "02-20-2018.csv",
            "02-21-2018.csv",
            "02-22-2018.csv",
            "02-23-2018.csv",
            "03-01-2018.csv",
            "03-02-2018.csv",
        ],
        "sample_fraction": 0.10,
        "min_samples_per_class": 5000,
        "description": "CIC-IDS2018 — 16M+ flows, 14 attack types, CIC Canada",
        "kaggle_slug": "solarmainframe/ids-intrusion-csv",
    },
    "cicids2017": {
        "label_column": " Label",
        "attack_column": None,
        "csv_files": [
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
        ],
        "sample_fraction": 0.25,
        "min_samples_per_class": 10000,
        "description": "CICIDS2017 — 2.8M flows, 14 attack types, CIC Canada",
        "kaggle_slug": "cicdataset/cicids2017",
    },
}


def get_dataset_config(name: str = None) -> dict:
    """Get configuration for the specified dataset."""
    name = name or DATASET_NAME
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(DATASETS.keys())}")
    return DATASETS[name]


# Preprocessing (defaults — overridden per dataset)
CORRELATION_THRESHOLD = 0.85
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Models
MODELS = {
    "Random Forest": {"max_depth": 40},
    "Decision Tree": {},
    "KNN": {},
    "SVM": {},
    "AdaBoost": {"n_estimators": 50, "learning_rate": 1},
}

BEST_MODEL_NAME = "Random Forest"

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
