# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os

# You may change default values here or 
# set the corresponding environment variable.
default_paths = {
    "BOP_PATH": r"/path/to/bop/datasets",
    "BOP_RESULTS_PATH": r"/path/to/folder/with/results",
    "BOP_EVAL_PATH": r"/path/to/eval/folder",
    "BOP_OUTPUT_PATH": r"/path/to/output/folder",
    "BOP_RENDERER_PATH": r"/path/to/bop_renderer/build",
    "BOP_MESHLAB_PATH": r"/path/to/meshlabserver.exe",
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
num_workers = 10

# use torch to calculate the errors
use_gpu = False
