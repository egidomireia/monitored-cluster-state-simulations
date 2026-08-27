from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data_files"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SCALING_RESULTS_DIR = RESULTS_DIR / "finite_size_scaling"

AUTOSCALE_SCRIPT = (
    PROJECT_ROOT
    / "external"
    / "autoscale"
    / "autoScale.py"
)