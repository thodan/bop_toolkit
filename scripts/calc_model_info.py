# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates the 3D bounding box and the diameter of 3D object models."""
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "hot3d",
    # Type of input object models.
    "model_type": None,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
}
################################################################################


# Load dataset parameters.
dp_model = dataset_params.get_model_params(
    p["datasets_path"], p["dataset"], p["model_type"]
)

models_info = {}
for obj_id in dp_model["obj_ids"]:
    misc.log("Processing model of object {}...".format(obj_id))

    model = inout.load_ply(dp_model["model_tpath"].format(obj_id=obj_id))

    # Calculate 3D bounding box.
    xs, ys, zs = model["pts"][:,0], model["pts"][:,1], model["pts"][:,2]
    bbox = misc.calc_3d_bbox(xs, ys, zs)

    # Calculated diameter.
    diameter = misc.calc_pts_diameter(model["pts"])

    models_info[obj_id] = {
        "min_x": bbox[0],
        "min_y": bbox[1],
        "min_z": bbox[2],
        "size_x": bbox[3],
        "size_y": bbox[4],
        "size_z": bbox[5],
        "diameter": diameter,
    }

# Save the calculated info about the object models.
inout.save_json(dp_model["models_info_path"], models_info)
