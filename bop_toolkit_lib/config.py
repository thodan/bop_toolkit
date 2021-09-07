import os


######## Basic ########

# Folder with the BOP datasets.
if 'BOP_PATH' in os.environ:
  datasets_path = os.environ['BOP_PATH']
else:
  datasets_path = r'/path/to/bop/datasets'


# Folder for the calculated pose errors and performance scores.
eval_path = r'/home_local/sund_ma/src/foreign_packages/bop/bop_results/bop_challenge_2019_eval'

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/home_local/sund_ma/src/foreign_packages/bop/my_util_scripts/eval'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'
