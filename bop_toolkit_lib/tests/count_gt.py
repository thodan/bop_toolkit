# Author: Van Nguyen Nguyen (van-nguyen.nguyen@enpc.fr)
# IMAGINE team, ENPC, France

"""Generating estimation from GT for debugging/unit tests purposes."""

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
import os

# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "tless",
    # Dataset split. Options: 'train', 'test'.
    "dataset_split": "test",
    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Minimum visibility of the GT poses to include them in the output.
    "min_visib_gt": 0.0,
}
################################################################################

datasets_path = p["datasets_path"]
dataset_name = p["dataset"]
split = p["dataset_split"]
split_type = p["dataset_split_type"]
min_visib_gt = p["min_visib_gt"]

dp_split = dataset_params.get_split_params(
    datasets_path, dataset_name, split, split_type=split_type
)
dp_model = dataset_params.get_model_params(datasets_path, dataset_name)

complete_split = split
if dp_split["split_type"] is not None:
    complete_split += "_" + dp_split["split_type"]


model_type = "eval"
dp_model = dataset_params.get_model_params(p["datasets_path"], dataset_name, model_type)

# Load info about the object models.
models_info = inout.load_json(dp_model["models_info_path"], keys_to_int=True)

# Load the estimation targets to consider.
targets = inout.load_json(os.path.join(dp_split["base_path"], "test_targets_bop19.json"))

num_instances = 0
for target in targets:
    num_instances += target["inst_count"]
print(f"Number of instances: {num_instances}")