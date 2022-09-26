# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os
import yaml


######## Basic ########

# Folder with the BOP datasets.
if 'BOP_PATH' in os.environ:
  datasets_path = os.environ['BOP_PATH']
else:
  datasets_path = r'/path/to/bop/datasets'

# Folder with pose results to be evaluated.
results_path = r'/path/to/folder/with/results'

# Folder for the calculated pose errors and performance scores.
eval_path = r'/path/to/eval/folder'

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/path/to/output/folder'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'

# Class for loading yaml as an object
class ParamLoader(object):
  def __init__(self, config):
    # Load from file path
    if type(config) is str:
      with open(config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    for name, value in config.items():
      setattr(self, name, self._wrap(value))

  def _wrap(self, value):
    if isinstance(value, (tuple, list, set, frozenset)): 
      return type(value)([self._wrap(v) for v in value])
    else:
      return ParamLoader(value) if isinstance(value, dict) else value