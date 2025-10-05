# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models under all identified symmetry transformations."""

import os
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import transform as tr


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "xyzibd",
    # Type of the renderer (used for the VSD pose error function).
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    # See misc.get_symmetry_transformations().
    "max_sym_disc_step": 0.01,
    "views": [
        {
            "R": tr.rotation_matrix(0.5 * np.pi, [1, 0, 0])
            .dot(tr.rotation_matrix(-0.5 * np.pi, [0, 0, 1]))
            .dot(tr.rotation_matrix(0.1 * np.pi, [0, 1, 0]))[:3, :3],
            "t": np.array([[0, 0, 500]]).T,
        }
    ],
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Folder for output visualisations.
    "vis_path": os.path.join(config.output_path, "vis_object_symmetries"),
    # Path templates for output images.
    "vis_rgb_tpath": os.path.join(
        "{vis_path}", "{dataset}", "{obj_id:06d}", "{view_id:06d}_{pose_id:06d}.jpg"
    ),
}
################################################################################


# Load dataset parameters.
model_type = None  # None = default.
if p["dataset"] == "tless":
    model_type = "cad"
dp_model = dataset_params.get_model_params(p["datasets_path"], p["dataset"], model_type)

# Use reasonable camera intrinsics default for rendering (copied from T-LESS)
width, height = 1280, 1024
fx, fy, cx, cy = 1075, 1073, 641, 507

# Create a renderer.
ren = renderer.create_renderer(
    width, height, p["renderer_type"], mode="rgb", shading="flat"
)

# Load meta info about the models (including symmetries).
models_info = inout.load_json(dp_model["models_info_path"], keys_to_int=True)


for obj_id in dp_model["obj_ids"]:
    # Load object model.
    misc.log("Loading 3D model of object {}...".format(obj_id))
    model_path = dp_model["model_tpath"].format(obj_id=obj_id)
    ren.add_object(obj_id, model_path)

    poses = misc.get_symmetry_transformations(
        models_info[obj_id], p["max_sym_disc_step"]
    )

    for pose_id, pose in enumerate(poses):
        for view_id, view in enumerate(p["views"]):
            R = view["R"].dot(pose["R"])
            t = view["R"].dot(pose["t"]) + view["t"]

            vis_rgb = ren.render_object(obj_id, R, t, fx, fy, cx, cy)["rgb"]

            # Path to the output RGB visualization.
            vis_rgb_path = p["vis_rgb_tpath"].format(
                vis_path=p["vis_path"],
                dataset=p["dataset"],
                obj_id=obj_id,
                view_id=view_id,
                pose_id=pose_id,
            )
            misc.ensure_dir(os.path.dirname(vis_rgb_path))
            inout.save_im(vis_rgb_path, vis_rgb)

misc.log("Done.")
