# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models in pose estimates saved in the BOP format."""

import os
import argparse
import numpy as np
import itertools

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
    # Top N pose estimates (with the highest score) to be visualized for each
    # object in each image.
    "n_top": 0,  # 0 = all estimates, -1 = given by the number of GT poses.
    # True = one visualization for each (im_id, obj_id), False = one per im_id.
    "vis_per_obj_id": True,
    # Indicates whether to render RGB image.
    "vis_rgb": True,
    # Indicates whether to resolve visibility in the rendered RGB images (using
    # depth renderings). If True, only the part of object surface, which is not
    # occluded by any other modeled object, is visible. If False, RGB renderings
    # of individual objects are blended together.
    "vis_rgb_resolve_visib": True,
    # Indicates whether to render depth image.
    "vis_depth_diff": True,
    # If to use the original model color.
    "vis_orig_color": False,
    # Type of the renderer (used for the VSD pose error function).
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    # Names of files with pose estimates to visualize (assumed to be stored in
    # folder config.eval_path). See docs/bop_challenge_2019.md for a description
    # of the format. Example results can be found at:
    # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip
    "result_filenames": [
        "/path/to/csv/with/results",
    ],
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Folder for output visualisations.
    "vis_path": os.path.join(config.output_path, "vis_est_poses"),
}
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--n_top", type=int, default=p["n_top"])
parser.add_argument("--vis_per_obj_id", type=bool, default=p["vis_per_obj_id"])
parser.add_argument("--vis_rgb", type=bool, default=p["vis_rgb"])
parser.add_argument("--vis_rgb_resolve_visib", type=bool, default=p["vis_rgb_resolve_visib"])
parser.add_argument("--vis_depth_diff", type=bool, default=p["vis_depth_diff"])
parser.add_argument("--vis_orig_color", type=bool, default=p["vis_orig_color"])
parser.add_argument("--renderer_type", type=str, default=p["renderer_type"])
parser.add_argument(
    "--result_filenames",
    type=str,
    default=",".join(p["result_filenames"]),
    help="Comma-separated names of files with results.",
)
parser.add_argument("--results_path", type=str, default=p["results_path"])
parser.add_argument("--datasets_path", type=str, default=p["datasets_path"])
parser.add_argument("--vis_path", type=str, default=p["vis_path"])
args = parser.parse_args()

result_filenames = args.result_filenames.split(",")

# Load colors.
colors_path = os.path.join(os.path.dirname(visualization.__file__), "colors.json")
colors = inout.load_json(colors_path)


# Path templates for output images.
vis_rgb_tpath = os.path.join(
    "{vis_path}", "{result_name}", "{scene_id:06d}", "{vis_name}.jpg"
)
vis_depth_diff_tpath = os.path.join(
    "{vis_path}", "{result_name}", "{scene_id:06d}", "{vis_name}_depth_diff.jpg"
)

for result_filename in result_filenames:
    misc.log("Processing: " + result_filename)

    # Parse info about the method and the dataset from the filename.
    result_name, method, dataset, split, split_type, _ = inout.parse_result_filename(result_filename)

    #######################
    # hot3d specific checks
    if dataset == "hot3d" and not htt_available:
        raise ImportError("Missing hand_tracking_toolkit dependency, mandatory for HOT3D dataset.")

    if dataset == "hot3d" and args.renderer_type != "htt":
        raise ValueError("'htt' renderer_type is mandatory for HOT3D dataset.")

    # hot3d does not contain depth modality, some visualizations are not available
    if dataset in ["hot3d"]:
        args.vis_rgb = True
        args.vis_rgb_resolve_visib = False
        args.vis_depth_diff = False
    #######################

    # Load dataset parameters.
    dp_split = dataset_params.get_split_params(
        args.datasets_path, dataset, split, split_type
    )

    model_type = "eval"
    dp_model = dataset_params.get_model_params(args.datasets_path, dataset, model_type)

    # Load pose estimates.
    misc.log("Loading pose estimates...")
    ests = inout.load_bop_results(os.path.join(args.results_path, result_filename))

    # Organize the pose estimates by scene, image and object.
    misc.log("Organizing pose estimates...")
    ests_org = {}
    for est in ests:
        ests_org.setdefault(est["scene_id"], {}).setdefault(
            est["im_id"], {}
        ).setdefault(est["obj_id"], []).append(est)

    # Rendering mode.
    renderer_modalities = []
    if args.vis_rgb:
        renderer_modalities.append("rgb")
    if args.vis_depth_diff or (args.vis_rgb and args.vis_rgb_resolve_visib):
        renderer_modalities.append("depth")
    renderer_mode = "+".join(renderer_modalities)

    width, height = None, None
    ren = None

    for scene_id, scene_ests in ests_org.items():
        tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)
        scene_modality = dataset_params.get_scene_sensor_or_modality(dp_split["eval_modality"], scene_id)
        scene_sensor = dataset_params.get_scene_sensor_or_modality(dp_split["eval_sensor"], scene_id)

        # Create a new renderer if image size has changed
        scene_width, scene_height = dataset_params.get_im_size(dp_split, scene_modality, scene_sensor)
        if (width, height) != (scene_width, scene_height):
            width, height = scene_width, scene_height
            misc.log(f"Creating renderer of type {p['renderer_type']}")
            ren = renderer.create_renderer(
                width, height, args.renderer_type, mode=renderer_mode, shading="flat"
            )
            # Load object models in the new renderer.
            for obj_id in dp_model["obj_ids"]:
                misc.log(f"Loading 3D model of object {obj_id}...")
                model_path = dp_model["model_tpath"].format(obj_id=obj_id)
                model_color = None
                if not args.vis_orig_color:
                    model_color = tuple(colors[(obj_id - 1) % len(colors)])
                ren.add_object(obj_id, model_path, surf_color=model_color)


        # Load info and ground-truth poses for the current scene.
        scene_camera = inout.load_scene_camera(dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id))
        scene_gt = inout.load_scene_gt(dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id))

        for im_ind, (im_id, im_ests) in enumerate(scene_ests.items()):
            if im_ind % 10 == 0:
                split_type_str = " - " + split_type if split_type is not None else ""
                misc.log(f"Visualizing pose estimates - method: {method}, dataset: {dataset}{split_type_str}, scene: {scene_id}, im: {im_id}")

            # Retrieve camera intrinsics.
            if dataset == 'hot3d':
                cam = pose_error_htt.create_camera_model(scene_camera[im_id])
            else:
                cam = scene_camera[im_id]["cam_K"]

            im_ests_vis = []
            im_ests_vis_obj_ids = []
            for obj_id, obj_ests in im_ests.items():
                # Sort the estimates by score (in descending order).
                obj_ests_sorted = sorted(
                    obj_ests, key=lambda est: est["score"], reverse=True
                )

                # Select the number of top estimated poses to visualize.
                if args.n_top == 0:  # All estimates are considered.
                    n_top_curr = None
                elif args.n_top == -1:  # Given by the number of GT poses.
                    n_gt = sum([gt["obj_id"] == obj_id for gt in scene_gt[im_id]])
                    n_top_curr = n_gt
                else:  # Specified by the parameter n_top.
                    n_top_curr = args.n_top
                obj_ests_sorted = obj_ests_sorted[slice(0, n_top_curr)]

                # Get list of poses to visualize.
                for est in obj_ests_sorted:
                    est["obj_id"] = obj_id

                    # Text info to write on the image at the pose estimate.
                    if args.vis_per_obj_id:
                        est["text_info"] = [
                            {"name": "", "val": est["score"], "fmt": ":.2f"}
                        ]
                    else:
                        val = "{}:{:.2f}".format(obj_id, est["score"])
                        est["text_info"] = [{"name": "", "val": val, "fmt": ""}]

                im_ests_vis.append(obj_ests_sorted)
                im_ests_vis_obj_ids.append(obj_id)

            # Join the per-object estimates if only one visualization is to be made.
            if not args.vis_per_obj_id:
                im_ests_vis = [list(itertools.chain.from_iterable(im_ests_vis))]

            for ests_vis_id, ests_vis in enumerate(im_ests_vis):
                # Load the color and depth images and prepare images for rendering.
                rgb = None
                if args.vis_rgb:
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
                if args.vis_depth_diff or (args.vis_rgb and args.vis_rgb_resolve_visib):
                    depth_available = dataset_params.sensor_has_modality(dp_split, scene_sensor, "depth")
                    if not depth_available:
                        misc.log(f"{scene_sensor} has no depth data, skipping depth visualization")
                        args.vis_depth_diff = False
                        args.vis_rgb_resolve_visib = False
                    else:
                        depth = inout.load_depth(
                            dp_split[tpath_keys["depth_tpath"]].format(scene_id=scene_id, im_id=im_id)
                        )
                        depth *= scene_camera[im_id]["depth_scale"]  # Convert to [mm].

                # Visualization name.
                if args.vis_per_obj_id:
                    vis_name = "{im_id:06d}_{obj_id:06d}".format(
                        im_id=im_id, obj_id=im_ests_vis_obj_ids[ests_vis_id]
                    )
                else:
                    vis_name = "{im_id:06d}".format(im_id=im_id)

                # Path to the output RGB visualization.
                vis_rgb_path = None
                if args.vis_rgb:
                    vis_rgb_path = vis_rgb_tpath.format(
                        vis_path=args.vis_path,
                        result_name=result_name,
                        scene_id=scene_id,
                        vis_name=vis_name,
                    )

                # Path to the output depth difference visualization.
                vis_depth_diff_path = None
                if args.vis_depth_diff:
                    vis_depth_diff_path = vis_depth_diff_tpath.format(
                        vis_path=args.vis_path,
                        result_name=result_name,
                        scene_id=scene_id,
                        vis_name=vis_name,
                    )

                # Visualization.
                visualization.vis_object_poses(
                    poses=ests_vis,
                    K=cam,
                    renderer=ren,
                    rgb=rgb,
                    depth=depth,
                    vis_rgb_path=vis_rgb_path,
                    vis_depth_diff_path=vis_depth_diff_path,
                    vis_rgb_resolve_visib=args.vis_rgb_resolve_visib,
                )

misc.log("Done.")
