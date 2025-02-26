# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates masks of object models in the ground-truth poses."""

import os
import argparse
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visibility


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "xyzibd",
    # Dataset split. Options: 'train', 'val', 'test'.
    "dataset_split": "test",
    # Dataset split type. None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Predefined test targets, either 'test_targets_bop19.json' or 'test_targets_bop24.json'. 
    "targets_filename": "test_targets_bop19.json",
    # Instead of using the predefined test targets, use all GT poses.
    "use_all_gt": True,
    # Tolerance used in the visibility test [mm].
    "delta": 15,  # 5 for ITODD, 15 for the other datasets.
    # Type of the renderer.
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # which modality to compute masks on, default to eval modality
    "modality": None,
    # which sensor to compute masks on, default to eval sensor
    "sensor": None,
}
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=p["dataset"])
parser.add_argument("--dataset_split", default=p["dataset_split"])
parser.add_argument("--dataset_split_type", default=p["dataset_split_type"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
parser.add_argument("--use_all_gt", action="store_true", default=p["use_all_gt"])
parser.add_argument("--dont_use_all_gt", action="store_false", dest="use_all_gt")
parser.add_argument("--delta", default=p["delta"])
parser.add_argument("--renderer_type", default=p["renderer_type"])
parser.add_argument("--datasets_path", default=p["datasets_path"])
args = parser.parse_args()


split_type = None if args.dataset_split_type is None else str(args.dataset_split_type)


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    args.datasets_path, args.dataset, args.dataset_split, split_type
)
if p["modality"] is None:
    p["modality"] = dp_split["eval_modality"]
if p["sensor"] is None:
    p["sensor"] = dp_split["eval_sensor"]

model_type = None
if args.dataset == "tless":
    model_type = "cad"
dp_model = dataset_params.get_model_params(args.datasets_path, args.dataset, model_type)

# Load and organize the estimation targets.
target_file_path = os.path.join(dp_split["base_path"], p["targets_filename"])
targets = inout.load_json(target_file_path)
targets_org = misc.reorganize_targets(targets)

scene_ids = dataset_params.get_present_scene_ids(dp_split)
for scene_id in scene_ids:
    tpath_keys = dataset_params.scene_tpaths_keys(p["modality"], p["sensor"], scene_id)

    # Load scene GT.
    scene_camera_path = dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id)
    scene_camera = inout.load_scene_camera(scene_camera_path)
    scene_gt_path = dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id)
    scene_gt = inout.load_scene_gt(scene_gt_path)

    # Create folders for the output masks (if they do not exist yet).
    mask_dir_path = os.path.dirname(
        dp_split[tpath_keys["mask_tpath"]].format(scene_id=scene_id, im_id=0, gt_id=0)
    )
    misc.log(f"Saving masks in {mask_dir_path}")
    misc.ensure_dir(mask_dir_path)
    mask_visib_dir_path = os.path.dirname(
        dp_split[tpath_keys["mask_visib_tpath"]].format(scene_id=scene_id, im_id=0, gt_id=0)
    )
    misc.log(f"Saving visible masks in {mask_visib_dir_path}")
    misc.ensure_dir(mask_visib_dir_path)

    # Initialize a renderer.
    misc.log("Initializing renderer...")
    if isinstance(dp_split["im_size"], dict):  
        width, height = dp_split["im_size"][p["sensor"]]
    else: # classical BOP format
        width, height = dp_split["im_size"]
    ren = renderer.create_renderer(
        width, height, renderer_type=args.renderer_type, mode="depth"
    )

    # Add object models.
    for obj_id in dp_model["obj_ids"]:
        ren.add_object(obj_id, dp_model["model_tpath"].format(obj_id=obj_id))

    for im_id in scene_gt:
        # Skip if the image is not in the targets
        in_target = scene_id in targets_org and im_id in targets_org[scene_id]
        if not args.use_all_gt and not in_target:
            misc.log("Skip image {} in scene {}".format(im_id, scene_id))
            continue

        misc.log(
            f"Calculating mask - dataset: {args.dataset} "
            f"({args.dataset_split}, {split_type}), scene: {scene_id}, im: {im_id}"
        )

        K = scene_camera[im_id]["cam_K"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # Load depth image.
        depth_fpath = dp_split[tpath_keys["depth_tpath"]].format(scene_id=scene_id, im_id=im_id)
        if not os.path.exists(depth_fpath):
            depth_fpath = depth_fpath.replace(".tif", ".png")        
        depth_im = inout.load_depth(depth_fpath)
        depth_im *= scene_camera[im_id]["depth_scale"]  # to [mm]
        dist_im = misc.depth_im_to_dist_im_fast(depth_im, K)

        for gt_id, gt in enumerate(scene_gt[im_id]):
            # Render the depth image.
            depth_gt = ren.render_object(
                gt["obj_id"], gt["cam_R_m2c"], gt["cam_t_m2c"], fx, fy, cx, cy
            )["depth"]

            # Convert depth image to distance image.
            dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)

            # Mask of the full object silhouette.
            mask = dist_gt > 0

            # Mask of the visible part of the object silhouette.
            mask_visib = visibility.estimate_visib_mask_gt(
                dist_im, dist_gt, args.delta, visib_mode="bop19"
            )

            # Save the calculated masks.
            mask_path = dp_split[tpath_keys["mask_tpath"]].format(
                scene_id=scene_id, im_id=im_id, gt_id=gt_id
            )
            inout.save_im(mask_path, 255 * mask.astype(np.uint8))

            mask_visib_path = dp_split[tpath_keys["mask_visib_tpath"]].format(
                scene_id=scene_id, im_id=im_id, gt_id=gt_id
            )
            inout.save_im(mask_visib_path, 255 * mask_visib.astype(np.uint8))
