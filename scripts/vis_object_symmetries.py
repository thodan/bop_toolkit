# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models under all identified symmetry transformations."""

import os
import math
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import trimesh

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import transform as tr
<<<<<<< HEAD
from bop_toolkit_lib import pycoco_utils
=======
from bop_toolkit_lib.rendering import renderer
>>>>>>> master


def generate_candidate_orientations(dist):
    views = [
        np.eye(4),
        tr.rotation_matrix( math.pi/2, [1, 0, 0]),
        tr.rotation_matrix(-math.pi/2, [1, 0, 0]),
        tr.rotation_matrix( math.pi/2, [0, 1, 0]),
        tr.rotation_matrix(-math.pi/2, [0, 1, 0]),
        tr.rotation_matrix( math.pi/2, [0, 0, 1]),
        tr.rotation_matrix(-math.pi/2, [0, 0, 1]),
    ]
    for view in views:
        view[2, 3] = dist

    return [{"R": v[:3,:3], "t": v[:3,3].reshape((3,1))} for v in views]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="xyzibd", help="Name of the BOP dataset. See dataset_params.py for options.")
parser.add_argument("--model_type", default=None, help="Type of the 3D models to use. None = default")
parser.add_argument("--renderer_type", default="vispy", help="Type of the renderer. Options: 'vispy', 'cpp', 'python'")
parser.add_argument("--max_sym_disc_step", type=float, default=0.01, help="See misc.get_symmetry_transformations")
parser.add_argument("--datasets_path", default=config.datasets_path, help="Path to the folder containing the BOP datasets.")
parser.add_argument("--vis_dir", default=os.path.join(config.output_path, "vis_object_symmetries"), help="Path to the folder for output visualisations.")
parser.add_argument("--render_res", type=int, default=640, help="Resolution of the symmetry renders.")
parser.add_argument("--fov_deg", type=int, default=60, help="FOV of the camera used for renders (degrees).")
args = parser.parse_args()

vis_dir = Path(args.vis_dir)
vis_dir.mkdir(parents=True, exist_ok=True)

# Load dataset parameters.
model_type = args.model_type
if model_type is None and args.dataset == "tless":
    model_type = "cad"
dp_model = dataset_params.get_model_params(args.datasets_path, args.dataset, model_type)

# generate some reasonable camera intrinsics
fov_rad = args.fov_deg * math.pi / 180.0
cx, cy = args.render_res / 2, args.render_res / 2
fx = args.render_res / 2 / math.tan(fov_rad / 2)
fy = fx

# Load meta info about the models (including symmetries).
models_info = inout.load_json(dp_model["models_info_path"], keys_to_int=True)
symmetry_poses = {obj_id: misc.get_symmetry_transformations(models_info[obj_id], args.max_sym_disc_step)
                  for obj_id in dp_model["obj_ids"]}


# Create a renderer and add objects.
ren = renderer.create_renderer(
    args.render_res, args.render_res, args.renderer_type, mode="rgb", shading="flat"
)
obj_models = {}
for obj_id in tqdm(dp_model["obj_ids"], desc="Adding objects to the renderer"):
    model_path = dp_model["model_tpath"].format(obj_id=obj_id)
    obj_models[obj_id] = trimesh.load(model_path, force="mesh")
    ren.add_object(obj_id, model_path)


# Set up the renderer camera at fixed distance to objects
# so that the biggest object fits in the view.
max_vertex_distance_from_origin = max(np.max(np.linalg.norm(obj_models[obj_id].vertices, axis=1))
                                      for obj_id in dp_model["obj_ids"])
diameter_pix = 0.99 * args.render_res  # portion of the image covered by the biggest object
distance = 2*max_vertex_distance_from_origin * fx / diameter_pix
views = generate_candidate_orientations(distance)

# select the best view for each object: maximize visible surface area
best_views = {}
for obj_id in tqdm(dp_model["obj_ids"], desc="Selecting best views"):
    best_view_id, best_bbox = None, -1
    for view_id, view in enumerate(views):
        rgb = ren.render_object(obj_id, view["R"], view["t"], fx, fy, cx, cy)["rgb"]
        mask = np.any(rgb > 0, axis=2)
        x, y, w, h = pycoco_utils.bbox_from_binary_mask(mask)
        bbox_area = w * h
        if bbox_area > best_bbox:
            best_view_id, best_surface = view_id, bbox_area
    best_views[obj_id] = views[best_view_id]

<<<<<<< HEAD
# Render objects under all symmetry transformations.
for obj_id in tqdm(dp_model["obj_ids"], desc="Rendering object symmetries"):
    view = best_views[obj_id]

    for pose_id, pose_s in enumerate(symmetry_poses[obj_id]):
        # T_cam_obj = T_view * T_symmetry
        R = view["R"].dot(pose_s["R"])
        t = view["R"].dot(pose_s["t"]) + view["t"]
        vis_rgb = ren.render_object(obj_id, R, t, fx, fy, cx, cy)["rgb"]
        vis_rgb_path = vis_dir / args.dataset / f"{obj_id:06d}" / f"{view_id:06d}_{pose_id:06d}.jpg"
        vis_rgb_path.parent.mkdir(parents=True, exist_ok=True)
        inout.save_im(vis_rgb_path, vis_rgb)

misc.log(f"Saved all symmetries in {vis_dir / args.dataset}")
=======
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
            misc.log(vis_rgb_path)
misc.log("Done.")
>>>>>>> master
