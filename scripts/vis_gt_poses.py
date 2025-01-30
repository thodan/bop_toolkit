# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""
Visualizes object models in the ground-truth poses.
The script visualize datasets in the classical BOP19 format as well as the HOT3D dataset in H3 BOP24 format.
"""

import os
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visualization

# Get the base name of the file without the .py extension
file_name = os.path.splitext(os.path.basename(__file__))[0]
logger = misc.get_logger(file_name)

htt_available = False
try:
    from bop_toolkit_lib import pose_error_htt
    htt_available = True
except ImportError as e:
    logger.warning("""Missing hand_tracking_toolkit dependency,
                   mandatory if you are running evaluation on HOT3d.
                   Refer to the README.md for installation instructions.
                   """)


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "xyzibd",
    # Dataset split. Options: 'train', 'val', 'test'.
    "dataset_split": "test",
    # Dataset split type. None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # File with a list of estimation targets used to determine the set of images
    # for which the GT poses will be visualized. The file is assumed to be stored
    # in the dataset folder. None = all images.
    # 'targets_filename': 'test_targets_bop19.json',
    "targets_filename": None,
    # Select ID's of scenes, images and GT poses to be processed.
    # Empty list [] means that all ID's will be used.
    "scene_ids": [],
    "im_ids": [],
    "gt_ids": [],
    #########
    # Which sensor to visualize, . By default it uses the evaluation modality set
    # in dataset_params.py. Set to None for rendering PBR images or BOP core datasets.
    # Set to sensor for new BOP core sets, e.g. "photoneo".
    #########
    # Modality used to visualize ground truth, default to eval modality. Should not be "depth".
    "modality": None,
    # Sensor used to visualize ground truth, default to eval sensor.
    "sensor": None,

    # ---------------------------------------------------------------------------------
    # Next parameters apply only to dataset with aligned color and depth images.
    # ---------------------
    # Indicates whether to render RGB images.
    "vis_rgb": True,
    # Indicates whether to resolve visibility in the rendered RGB images (using
    # depth renderings). If True, only the part of object surface, which is not
    # occluded by any other modeled object, is visible. If False, RGB renderings
    # of individual objects are blended together.
    "vis_rgb_resolve_visib": True,
    # Indicates whether to save images of depth differences.
    "vis_depth_diff": True,
    # ---------------------------------------------------------------------------------

    # Whether to use the original model color.
    "vis_orig_color": True,
    # Type of the renderer (used for the VSD pose error function).
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'. 'htt' is mandatory for "hot3d" dataset.
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Folder for output visualisations.
    "vis_path": os.path.join(config.output_path, "vis_gt_poses"),
    # Path templates for output images.
    "vis_rgb_tpath": os.path.join(
        "{vis_path}", "{dataset}", "{split}", "{scene_id:06d}", "{im_id:06d}.jpg"
    ),
    "vis_depth_diff_tpath": os.path.join(
        "{vis_path}",
        "{dataset}",
        "{split}",
        "{scene_id:06d}",
        "{im_id:06d}_depth_diff.jpg",
    ),
}
################################################################################

#######################
# hot3d specific checks
if p["dataset"] == "hot3d" and not htt_available:
    raise ImportError("Missing hand_tracking_toolkit dependency, mandatory for HOT3D dataset.")

if p["dataset"] == "hot3d" and p["renderer_type"] != "htt":
    raise ValueError("'htt' renderer_type is mandatory for HOT3D dataset.")

# hot3d does not contain depth modality, some visualizations are not available
if p["dataset"] in ["hot3d"]:
    p["vis_rgb"] = True
    p["vis_rgb_resolve_visib"] = False
    p["vis_depth_diff"] = False
#######################

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], p["dataset"], p["dataset_split"], p["dataset_split_type"]
)
if p["modality"] is None:
    p["modality"] = dp_split["eval_modality"]
assert p["modality"] != "depth", "Modality should be a color modality (not 'depth')"
if p["sensor"] is None:
    p["sensor"] = dp_split["eval_sensor"]

model_type = "eval"  # None = default.
dp_model = dataset_params.get_model_params(p["datasets_path"], p["dataset"], model_type)

# Load colors.
colors_path = os.path.join(os.path.dirname(visualization.__file__), "colors.json")
colors = inout.load_json(colors_path)

# Subset of images for which the ground-truth poses will be rendered.
if p["targets_filename"] is not None:
    targets = inout.load_json(
        os.path.join(dp_split["base_path"], p["targets_filename"])
    )
    scene_im_ids = {}
    for target in targets:
        scene_im_ids.setdefault(target["scene_id"], set()).add(target["im_id"])
else:
    scene_im_ids = None

# List of considered scenes.
scene_ids_curr = dp_split["scene_ids"]
if p["scene_ids"]:
    scene_ids_curr = set(scene_ids_curr).intersection(p["scene_ids"])

# Rendering mode.
renderer_modalities = []
if p["vis_rgb"]:
    renderer_modalities.append("rgb")
if p["vis_depth_diff"] or (p["vis_rgb"] and p["vis_rgb_resolve_visib"]):
    renderer_modalities.append("depth")
renderer_mode = "+".join(renderer_modalities)


width, height = None, None
ren = None

for scene_id in scene_ids_curr:
    tpath_keys = dataset_params.scene_tpaths_keys(p["modality"], p["sensor"], scene_id)
    scene_modality = dataset_params.get_scene_sensor_or_modality(p["modality"], scene_id)
    scene_sensor = dataset_params.get_scene_sensor_or_modality(p["sensor"], scene_id)

    # Create a new renderer if image size has changed
    scene_width, scene_height = dataset_params.get_im_size(dp_split, scene_modality, scene_sensor)
    if (width, height) != (scene_width, scene_height):
        width, height = scene_width, scene_height
        misc.log(f"Creating renderer of type {p['renderer_type']}")
        ren = renderer.create_renderer(
            width, height, p["renderer_type"], mode=renderer_mode, shading="flat"
        )
        # Load object models in the new renderer.
        for obj_id in dp_model["obj_ids"]:
            misc.log(f"Loading 3D model of object {obj_id}...")
            model_path = dp_model["model_tpath"].format(obj_id=obj_id)
            model_color = None
            if not p["vis_orig_color"]:
                model_color = tuple(colors[(obj_id - 1) % len(colors)])
            ren.add_object(obj_id, model_path, surf_color=model_color)

    # Load scene info and ground-truth poses.
    scene_camera = inout.load_scene_camera(dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id))
    scene_gt = inout.load_scene_gt(dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id))
    # List of considered images.
    if scene_im_ids is not None:
        im_ids = scene_im_ids[scene_id]
    else:
        im_ids = sorted(scene_gt.keys())
    if p["im_ids"]:
        im_ids = set(im_ids).intersection(p["im_ids"])

    # Render the object models in the ground-truth poses in the selected images.
    for im_counter, im_id in enumerate(im_ids):
        if im_counter % 10 == 0:
            misc.log(
                "Visualizing GT poses - dataset: {}, scene: {}, im: {}/{}".format(
                    p["dataset"], scene_id, im_counter, len(im_ids)
                )
            )

        # Retrieve camera intrinsics.
        if p['dataset'] == 'hot3d':
            cam = pose_error_htt.create_camera_model(scene_camera[im_id])
        else:
            cam = scene_camera[im_id]["cam_K"]

        # List of considered ground-truth poses.
        gt_ids_curr = range(len(scene_gt[im_id]))
        if p["gt_ids"]:
            gt_ids_curr = set(gt_ids_curr).intersection(p["gt_ids"])

        # Collect the ground-truth poses.
        gt_poses = []
        for gt_id in gt_ids_curr:
            gt = scene_gt[im_id][gt_id]
            # skip fully occluded masks - all values are -1
            if all(val == -1 for val in gt["cam_t_m2c"]):
                continue
            gt_poses.append(
                {
                    "obj_id": gt["obj_id"],
                    "R": gt["cam_R_m2c"],
                    "t": gt["cam_t_m2c"],
                    "text_info": [
                        {
                            "name": "",
                            "val": "{}:{}".format(gt["obj_id"], gt_id),
                            "fmt": "",
                        }
                    ],
                }
            )

        # Load the color and depth images and prepare images for rendering.
        rgb = None
        if p["vis_rgb"]:
            # rgb_tpath is an alias refering to the sensor|modality image paths on which the poses are rendered
            im_tpath = tpath_keys["rgb_tpath"]
            # check for BOP classic (itodd)
            rgb_available = dataset_params.sensor_has_modality(dp_split, scene_sensor, 'rgb')
            if im_tpath == "rgb_tpath" and not rgb_available:
                im_tpath = "gray_tpath"

            rgb = inout.load_im(
                dp_split[im_tpath].format(scene_id=scene_id, im_id=im_id)
            )
            # if image is grayscale (e.g. quest3), convert it to 3 channels
            if rgb.ndim == 2:
                rgb = np.dstack([rgb, rgb, rgb])
            else:
                rgb = rgb[:,:,:3]  # should we keep this?

        depth = None
        if p["vis_depth_diff"] or (p["vis_rgb"] and p["vis_rgb_resolve_visib"]):
            depth_available = dataset_params.sensor_has_modality(dp_split, scene_sensor, "depth")
            if not depth_available:
                misc.log(f"{scene_sensor} has no depth data, skipping depth visualization")
                p["vis_depth_diff"] = False
                p["vis_rgb_resolve_visib"] = False
            else:
                depth = inout.load_depth(
                    dp_split[tpath_keys["depth_tpath"]].format(scene_id=scene_id, im_id=im_id)
                )
                depth *= scene_camera[im_id]["depth_scale"]  # Convert to [mm].

        # Path to the output RGB visualization.
        split = "{}_{}".format(p["dataset_split"], scene_sensor) if scene_sensor else p["dataset_split"] 
        vis_rgb_path = None
        if p["vis_rgb"]:
            vis_rgb_path = p["vis_rgb_tpath"].format(
                vis_path=p["vis_path"],
                dataset=p["dataset"],
                split=split,
                scene_id=scene_id,
                im_id=im_id,
            )

        # Path to the output depth difference visualization.
        vis_depth_diff_path = None
        if p["vis_depth_diff"]:
            vis_depth_diff_path = p["vis_depth_diff_tpath"].format(
                vis_path=p["vis_path"],
                dataset=p["dataset"],
                split=split,
                scene_id=scene_id,
                im_id=im_id,
            )

        # Visualization.
        visualization.vis_object_poses(
            poses=gt_poses,
            K=cam,
            renderer=ren,
            rgb=rgb,
            depth=depth,
            vis_rgb_path=vis_rgb_path,
            vis_depth_diff_path=vis_depth_diff_path,
            vis_rgb_resolve_visib=p["vis_rgb_resolve_visib"],
        )

misc.log("Done.")
