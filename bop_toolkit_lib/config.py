# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT_DIR / "bop_toolkit_lib"
DATA_DIR = ROOT_DIR / "data"

# You may change default values here or 
# set the corresponding environment variable.
default_paths = {
    "BOP_PATH": str(DATA_DIR),
    "BOP_RESULTS_PATH": rf"{ROOT_DIR}/results",
    "BOP_EVAL_PATH": rf"{ROOT_DIR}/results",
    "BOP_OUTPUT_PATH": rf"{ROOT_DIR}/outputs",
    "BOP_RENDERER_PATH": rf"{ROOT_DIR}/bop_renderer/build",
    "BOP_MESHLAB_PATH": r"/path/to/meshlabserver.exe",
    "BOP_NUM_WORKERS": "8",
}


def get_env_default(env_var):
    """
    Return environment variable or default value secified in config.default_paths
    """
    return os.environ.get(env_var, default_paths[env_var])

######## Default paths ########

# Folder with the BOP datasets.
datasets_path = get_env_default("BOP_PATH")

# Folder with pose results to be evaluated.
results_path = get_env_default("BOP_RESULTS_PATH")

# Folder for the calculated pose errors and performance scores.
eval_path = get_env_default("BOP_EVAL_PATH")

# Folder for outputs (e.g. visualizations).
output_path = get_env_default("BOP_OUTPUT_PATH")

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = get_env_default("BOP_RENDERER_PATH")

# Executable of the MeshLab server.
meshlab_server_path = get_env_default("BOP_MESHLAB_PATH")

######## Other ########

# Number of workers for the parallel evaluation of pose errors.
num_workers = int(get_env_default("BOP_NUM_WORKERS"))

# use torch to calculate the errors
use_gpu = False
