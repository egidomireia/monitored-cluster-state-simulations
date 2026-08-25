from pathlib import Path
import sys

REPO_ROOT = next(
    folder for folder in (Path.cwd(), *Path.cwd().parents)
    if (folder / "src" / "monitored_cluster_states").is_dir()
)

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monitored_cluster_states.paths import RAW_DATA_DIR
from monitored_cluster_states.observables import initialize_data_q
from monitored_cluster_states.plotting import (
    plot_entropy_vs_L,
    plot_CV_vs_DV_fixed_L,
    plot_CV_vs_DV_with_inset,
)