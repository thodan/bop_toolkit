# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates visibility, 2D bounding boxes etc. for the ground-truth poses.

See docs/bop_datasets_format.md for documentation of the calculated info.

The info is saved in folder "{train,val,test}_gt_info" in the main folder of the
selected dataset.
"""

import os
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
    # Whether to save visualizations of visibility masks.
    "vis_visibility_masks": False,
    # Tolerance used in the visibility test [mm].
    "delta": 15,
    # Type of the renderer.
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # which modality to compute masks on, default to eval modality
    "modality": "rgb",
    # which sensor to compute masks on, default to eval sensor
    "sensor": "realsense",
    # Path template for output images with object masks.
    "vis_mask_visib_tpath": os.path.join(
        config.output_path,
        "vis_gt_visib_delta={delta}",
        "vis_gt_visib_delta={delta}",
        "{dataset}",
        "{split}",
        "{scene_id:06d}",
        "{im_id:06d}_{gt_id:06d}.jpg",
    ),
}
################################################################################


if p["vis_visibility_masks"]:
    from bop_toolkit_lib import visualization

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], p["dataset"], p["dataset_split"], p["dataset_split_type"]
)
if p["modality"] is None:
    p["modality"] = dp_split["eval_modality"]
if p["sensor"] is None:
    p["sensor"] = dp_split["eval_sensor"]

model_type = None
if p["dataset"] == "tless":
    model_type = "cad"
dp_model = dataset_params.get_model_params(p["datasets_path"], p["dataset"], model_type)

# Initialize a renderer.
misc.log("Initializing renderer...")

# The renderer has a larger canvas for generation of masks of truncated objects.
if isinstance(dp_split["im_size"], dict):  
    im_width, im_height = dp_split["im_size"][p["sensor"]]
else: # classical BOP format
    im_width, im_height = dp_split["im_size"]
ren_width, ren_height = 3 * im_width, 3 * im_height
ren_cx_offset, ren_cy_offset = im_width, im_height
ren = renderer.create_renderer(ren_width, ren_height, p["renderer_type"], mode="depth")

for obj_id in dp_model["obj_ids"]:
    model_fpath = dp_model["model_tpath"].format(obj_id=obj_id)
    ren.add_object(obj_id, model_fpath)

scene_ids = dataset_params.get_present_scene_ids(dp_split)
for scene_id in scene_ids:
    tpath_keys = dataset_params.scene_tpaths_keys(p["modality"], p["sensor"], scene_id)

    # Load scene GT.
    scene_camera_path = dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id)
    scene_camera = inout.load_scene_camera(scene_camera_path)
    scene_gt_path = dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id)
    scene_gt = inout.load_scene_gt(scene_gt_path)

    scene_gt_info = {}
    im_ids = sorted(scene_gt.keys())
    for im_counter, im_id in enumerate(im_ids):
        if im_counter % 100 == 0:
            misc.log(
                "Calculating GT info - dataset: {} ({}, {}), scene: {}, im: {}".format(
                    p["dataset"],
                    p["dataset_split"],
                    p["dataset_split_type"],
                    scene_id,
                    im_id,
                )
            )

        # Load depth image.
        depth_fpath = dp_split[tpath_keys["depth_tpath"]].format(scene_id=scene_id, im_id=im_id)
        if not os.path.exists(depth_fpath):
            depth_fpath = depth_fpath.replace(".tif", ".png")
        depth = inout.load_depth(depth_fpath)
        depth *= scene_camera[im_id]["depth_scale"]  # Convert to [mm].

        K = scene_camera[im_id]["cam_K"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        im_size = (depth.shape[1], depth.shape[0])

        scene_gt_info[im_id] = []
        for gt_id, gt in enumerate(scene_gt[im_id]):
            # Render depth image of the object model in the ground-truth pose.
            depth_gt_large = ren.render_object(
                gt["obj_id"],
                gt["cam_R_m2c"],
                gt["cam_t_m2c"],
                fx,
                fy,
                cx + ren_cx_offset,
                cy + ren_cy_offset,
            )["depth"]
            depth_gt = depth_gt_large[
                ren_cy_offset : (ren_cy_offset + im_height),
                ren_cx_offset : (ren_cx_offset + im_width),
            ]

            # Convert depth images to distance images.
            dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)
            dist_im = misc.depth_im_to_dist_im_fast(depth, K)

            # Estimation of the visibility mask.
            visib_gt = visibility.estimate_visib_mask_gt(
                dist_im, dist_gt, p["delta"], visib_mode="bop19"
            )

            # Mask of the object in the GT pose.
            obj_mask_gt_large = depth_gt_large > 0
            obj_mask_gt = dist_gt > 0

            # Number of pixels in the whole object silhouette
            # (even in the truncated part).
            px_count_all = np.sum(obj_mask_gt_large)

            # Number of pixels in the object silhouette with a valid depth measurement
            # (i.e. with a non-zero value in the depth image).
            px_count_valid = np.sum(dist_im[obj_mask_gt] > 0)

            # Number of pixels in the visible part of the object silhouette.
            px_count_visib = visib_gt.sum()

            # Visible surface fraction.
            if px_count_all > 0:
                visib_fract = px_count_visib / float(px_count_all)
            else:
                visib_fract = 0.0

            # Bounding box of the whole object silhouette
            # (including the truncated part).
            bbox = [-1, -1, -1, -1]
            if px_count_visib > 0:
                ys, xs = obj_mask_gt_large.nonzero()
                ys -= ren_cy_offset
                xs -= ren_cx_offset
                bbox = misc.calc_2d_bbox(xs, ys, im_size)

            # Bounding box of the visible surface part.
            bbox_visib = [-1, -1, -1, -1]
            if px_count_visib > 0:
                ys, xs = visib_gt.nonzero()
                bbox_visib = misc.calc_2d_bbox(xs, ys, im_size)

            # Store the calculated info.
            scene_gt_info[im_id].append(
                {
                    "px_count_all": int(px_count_all),
                    "px_count_valid": int(px_count_valid),
                    "px_count_visib": int(px_count_visib),
                    "visib_fract": float(visib_fract),
                    "bbox_obj": [int(e) for e in bbox],
                    "bbox_visib": [int(e) for e in bbox_visib],
                }
            )

            # Visualization of the visibility mask.
            if p["vis_visibility_masks"]:
                depth_im_vis = visualization.depth_for_vis(depth, 0.2, 1.0)
                depth_im_vis = np.dstack([depth_im_vis] * 3)

                visib_gt_vis = visib_gt.astype(np.float64)
                zero_ch = np.zeros(visib_gt_vis.shape)
                visib_gt_vis = np.dstack([zero_ch, visib_gt_vis, zero_ch])

                vis = 0.5 * depth_im_vis + 0.5 * visib_gt_vis
                vis[vis > 1] = 1

                vis_path = p["vis_mask_visib_tpath"].format(
                    delta=p["delta"],
                    dataset=p["dataset"],
                    split=p["dataset_split"],
                    scene_id=scene_id,
                    im_id=im_id,
                    gt_id=gt_id,
                )
                misc.ensure_dir(os.path.dirname(vis_path))
                inout.save_im(vis_path, vis)

    # Save the info for the current scene.
    scene_gt_info_path = dp_split[tpath_keys["scene_gt_info_tpath"]].format(scene_id=scene_id)
    misc.ensure_dir(os.path.dirname(scene_gt_info_path))
    inout.save_json(scene_gt_info_path, scene_gt_info)
