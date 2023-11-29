# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""'Uniformly' resamples and decimates 3D object models for evaluation.

Note: Models of some T-LESS objects were processed by Blender (using the Remesh
modifier).
"""

import os

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import misc


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "lm",
    # Type of input object models.
    # None = default model type.
    "model_in_type": None,
    # Type of output object models.
    "model_out_type": "eval",
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Path to meshlabserver.exe (tested version: 1.3.3).
    # On Windows: C:\Program Files\VCG\MeshLab133\meshlabserver.exe
    "meshlab_server_path": config.meshlab_server_path,
    # Path to scripts/meshlab_scripts/remesh_for_eval.mlx.
    "meshlab_script_path": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "meshlab_scripts",
        r"remesh_for_eval_cell=0.25.mlx",
    ),
}
################################################################################


# Load dataset parameters.
dp_model_in = dataset_params.get_model_params(
    p["datasets_path"], p["dataset"], p["model_in_type"]
)

dp_model_out = dataset_params.get_model_params(
    p["datasets_path"], p["dataset"], p["model_out_type"]
)

# Attributes to save for the output models.
attrs_to_save = []

# Process models of all objects in the selected dataset.
for obj_id in dp_model_in["obj_ids"]:
    misc.log("\n\n\nProcessing model of object {}...\n".format(obj_id))

    model_in_path = dp_model_in["model_tpath"].format(obj_id=obj_id)
    model_out_path = dp_model_out["model_tpath"].format(obj_id=obj_id)

    misc.ensure_dir(os.path.dirname(model_out_path))

    misc.run_meshlab_script(
        p["meshlab_server_path"],
        p["meshlab_script_path"],
        model_in_path,
        model_out_path,
        attrs_to_save,
    )

misc.log("Done.")
