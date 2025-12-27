"""
Visualizes object models in the GT/estimated poses.
The script visualize datasets in the classical BOP19 format as well as the HOT3D dataset in H3 BOP24 format.
"""

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
from bop_toolkit_lib import config, dataset_params, inout, misc, visualization
from bop_toolkit_lib.rendering import renderer
from tqdm import tqdm

file_name = os.path.splitext(os.path.basename(__file__))[0]
logger = misc.get_logger(file_name)

htt_available = False
try:
    from bop_toolkit_lib import pose_error_htt

    htt_available = True
except ImportError as e:
    logger.warning(
        """Missing hand_tracking_toolkit dependency,
        mandatory if you are running evaluation on HOT3d.
        Refer to the README.md for installation instructions.
        """
    )

DEFAULTS = {
    "common": {
        # Indicates whether to render RGB image.
        "vis_rgb": True,
        # Indicates whether to resolve visibility in the rendered RGB images (using
        # depth renderings). If True, only the part of object surface, which is not
        # occluded by any other modeled object, is visible. If False, RGB renderings
        # of individual objects are blended together.
        "vis_rgb_resolve_visib": True,
        # Indicates whether to render depth image (or save images of depth differences).
        "vis_depth_diff": True,
        # If to use the original model color.
        # Note: vis_gt_poses used True, vis_est_poses used False. Defaulting to False.
        "vis_orig_color": False,
        # Type of the renderer (used for the VSD pose error function).
        # Options: 'vispy', 'cpp', 'python'. 'htt' is mandatory for "hot3d" dataset.
        "renderer_type": "vispy",
        # Folder containing the BOP datasets.
        "datasets_path": config.datasets_path,
        # Folder for output visualisations.
        "vis_path": None,
    },
    "gt": {
        # See dataset_params.py for options.
        "dataset": "xyzbid",
        # Dataset split. Options: 'train', 'val', 'test'.
        "dataset_split": "test",
        # Dataset split type. None = default. See dataset_params.py for options.
        "dataset_split_type": None,
        # File with a list of estimation targets used to determine the set of images
        # for which the GT poses will be visualized. The file is assumed to be stored
        # in the dataset folder. None = all images.
        "targets_filename": "test_targets_bop19.json",
        # Modality used to visualize ground truth, default to eval modality. Should not be "depth".
        "modality": None,
        # Sensor used to visualize ground truth, default to eval sensor.
        "sensor": None,
        # Select ID's of scenes, images and GT poses to be processed.
        # Empty list [] means that all ID's will be used.
        "scene_ids": [],
        "im_ids": [],
        "gt_ids": [],
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
    },
    "est": {
        # Top N pose estimates (with the highest score) to be visualized for each
        # object in each image. 0 = all estimates, -1 = given by the number of GT poses.
        "n_top": 0,
        # Names of files with pose estimates to visualize (assumed to be stored in
        # folder config.eval_path).
        "result_filename": None,
        # Folder with results to be evaluated.
        "results_path": config.results_path,
        "vis_rgb_tpath": os.path.join(
            "{vis_path}", "{result_name}", "{scene_id:06d}", "{vis_name}.jpg"
        ),
        "vis_depth_diff_tpath": os.path.join(
            "{vis_path}", "{result_name}", "{scene_id:06d}", "{vis_name}_depth_diff.jpg"
        ),
    },
}
################################################################################


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Visualize object poses (Ground Truth or Estimated)"
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    c_defs = DEFAULTS["common"]

    misc.add_argument_bool(common_parser, "vis_rgb", c_defs["vis_rgb"])
    misc.add_argument_bool(
        common_parser, "vis_rgb_resolve_visib", c_defs["vis_rgb_resolve_visib"]
    )
    misc.add_argument_bool(common_parser, "vis_depth_diff", c_defs["vis_depth_diff"])
    misc.add_argument_bool(common_parser, "vis_orig_color", c_defs["vis_orig_color"])

    common_parser.add_argument(
        "--renderer_type",
        type=str,
        default=c_defs["renderer_type"],
        help="Renderer type (vispy, cpp, python, htt)",
    )
    common_parser.add_argument(
        "--datasets_path",
        type=str,
        default=c_defs["datasets_path"],
        help="Path to BOP datasets",
    )
    common_parser.add_argument(
        "--vis_path",
        type=str,
        default=c_defs["vis_path"],
        help="Output folder for visualizations",
    )

    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="Visualization mode"
    )

    gt_defs = DEFAULTS["gt"]
    parser_gt = subparsers.add_parser(
        "gt", parents=[common_parser], help="Visualize Ground Truth poses"
    )
    parser_gt.add_argument("--dataset", type=str, default=gt_defs["dataset"])
    parser_gt.add_argument(
        "--dataset_split", type=str, default=gt_defs["dataset_split"]
    )
    parser_gt.add_argument(
        "--dataset_split_type", type=str, default=gt_defs["dataset_split_type"]
    )
    parser_gt.add_argument(
        "--targets_filename",
        type=str,
        default=gt_defs["targets_filename"],
        help="JSON file with targets (scene/im_ids) to visualize",
    )
    parser_gt.add_argument("--modality", type=str, default=gt_defs["modality"])
    parser_gt.add_argument("--sensor", type=str, default=gt_defs["sensor"])

    parser_gt.add_argument("--scene_ids", type=str, help="Comma-separated scene IDs")
    parser_gt.add_argument("--im_ids", type=str, help="Comma-separated image IDs")
    parser_gt.add_argument("--gt_ids", type=str, help="Comma-separated GT IDs")

    parser_gt.add_argument(
        "--vis_rgb_tpath",
        type=str,
        default=gt_defs["vis_rgb_tpath"],
        help="Template path for output RGB images",
    )
    parser_gt.add_argument(
        "--vis_depth_diff_tpath",
        type=str,
        default=gt_defs["vis_depth_diff_tpath"],
        help="Template path for output depth difference images",
    )

    est_defs = DEFAULTS["est"]
    parser_est = subparsers.add_parser(
        "est", parents=[common_parser], help="Visualize Estimated poses"
    )
    parser_est.add_argument(
        "--n_top",
        type=int,
        default=est_defs["n_top"],
        help="Top N estimates to visualize (0=all, -1=match GT)",
    )
    parser_est.add_argument(
        "--result_filename",
        type=str,
        default=est_defs["result_filename"],
        help="Result file",
    )
    parser_est.add_argument(
        "--results_path",
        type=str,
        default=est_defs["results_path"],
        help="Path to results folder",
    )
    parser_est.add_argument(
        "--vis_rgb_tpath",
        type=str,
        default=est_defs["vis_rgb_tpath"],
        help="Template path for output RGB images",
    )
    parser_est.add_argument(
        "--vis_depth_diff_tpath",
        type=str,
        default=est_defs["vis_depth_diff_tpath"],
        help="Template path for output depth difference images",
    )
    return parser


def main(args):
    # Load colors.
    colors_path = os.path.join(os.path.dirname(visualization.__file__), "colors.json")
    colors = inout.load_json(colors_path)

    if args.mode == "gt":
        dataset, split, split_type = (
            args.dataset,
            args.dataset_split,
            args.dataset_split_type,
        )
        method = "GT"
    else:
        result_filename = args.result_filename

        misc.log("Processing: " + result_filename)

        # Parse info about the method and the dataset from the filename.
        result_name, method, dataset, split, split_type, _ = (
            inout.parse_result_filename(result_filename)
        )

    #######################
    # hot3d specific checks
    if dataset == "hot3d" and not htt_available:
        raise ImportError(
            "Missing hand_tracking_toolkit dependency, mandatory for HOT3D dataset."
        )

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

    if args.mode == "gt":
        # List of considered scenes.
        scene_ids_curr = dp_split["scene_ids"]
        if args.scene_ids:
            target_scene_ids = [int(x) for x in args.scene_ids.split(',')]
            scene_ids_curr = set(scene_ids_curr).intersection(target_scene_ids)
            if len(scene_ids_curr) == 0:
                misc.log(f"Dataset scene ids {dp_split['scene_ids']} do not overlap with chosen scene ids {args.scene_ids}")
        scene_ids = scene_ids_curr
                
        # Subset of images for which the ground-truth poses will be rendered.
        if args.targets_filename is not None:
            targets = inout.load_json(
                os.path.join(dp_split["base_path"], args.targets_filename)
            )
            scene_im_ids = {}
            for target in targets:
                scene_im_ids.setdefault(target["scene_id"], set()).add(target["im_id"])
        else:
            scene_im_ids = None
    else:
        # Load pose estimates.
        misc.log("Loading pose estimates...")
        ests = inout.load_bop_results(os.path.join(args.results_path, result_filename))

        # Organize the pose estimates by scene, image and object.
        misc.log("Organizing pose estimates...")
        ests_org = {}
        for est in ests:
            if est["scene_id"] not in [48]:
                continue
            ests_org.setdefault(est["scene_id"], {}).setdefault(
                est["im_id"], {}
            ).setdefault(est["obj_id"], []).append(est)
        scene_ids = list(ests_org.keys())

    # Rendering mode.
    renderer_modalities = []
    if args.vis_rgb:
        renderer_modalities.append("rgb")
    if args.vis_depth_diff or (args.vis_rgb and args.vis_rgb_resolve_visib):
        renderer_modalities.append("depth")
    renderer_mode = "+".join(renderer_modalities)

    width, height = None, None
    ren = None

    for scene_id in scene_ids:

        tpath_keys = dataset_params.scene_tpaths_keys(
            dp_split["eval_modality"], dp_split["eval_sensor"], scene_id
        )
        scene_modality = dataset_params.get_scene_sensor_or_modality(
            dp_split["eval_modality"], scene_id
        )
        scene_sensor = dataset_params.get_scene_sensor_or_modality(
            dp_split["eval_sensor"], scene_id
        )

        # Create a new renderer if image size has changed
        scene_width, scene_height = dataset_params.get_im_size(
            dp_split, scene_modality, scene_sensor
        )
        if (width, height) != (scene_width, scene_height):
            width, height = scene_width, scene_height
            misc.log(f"Creating renderer of type {args.renderer_type}")
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
        scene_camera = inout.load_scene_camera(
            dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id)
        )
        scene_gt = inout.load_scene_gt(
            dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id)
        )

        split_type_str = " - " + split_type if split_type is not None else ""
        misc.log(
            f"Visualizing pose estimates - method: {method}, dataset: {dataset}{split_type_str}, scene: {scene_id}"
        )

        if args.mode == "gt":
            # List of considered images.
            if scene_im_ids is not None:
                im_ids = scene_im_ids[scene_id]
            else:
                im_ids = sorted(scene_gt.keys())
            if args.im_ids:
                im_ids = set(im_ids).intersection(args.im_ids)
            poses_scene_vis = {}
            for im_counter, im_id in enumerate(im_ids):
                # List of considered ground-truth poses.
                gt_ids_curr = range(len(scene_gt[im_id]))
                if args.gt_ids:
                    gt_ids_curr = set(gt_ids_curr).intersection(args.gt_ids)

                # Collect the ground-truth poses.
                poses = []
                for gt_id in gt_ids_curr:
                    gt = scene_gt[im_id][gt_id]
                    # skip fully occluded masks - all values are -1
                    if all(val == -1 for val in gt["cam_t_m2c"]):
                        continue
                    poses.append(
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
                poses_scene_vis[im_id] = poses
        else:
            poses_scene = ests_org[scene_id]
            poses_scene_vis = {}
            for im_ind, (im_id, poses_img) in enumerate(poses_scene.items()):

                im_ests_vis = []
                im_ests_vis_obj_ids = []
                for obj_id, obj_ests in poses_img.items():
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
                        val = "{}:{:.2f}".format(obj_id, est["score"])
                        est["text_info"] = [{"name": "", "val": val, "fmt": ""}]

                    im_ests_vis.append(obj_ests_sorted)
                    im_ests_vis_obj_ids.append(obj_id)

                # Join the per-object estimates to make it a single visual.
                # if there are multiple estimates per object, they are treated as independent entries
                poses = list(itertools.chain.from_iterable(im_ests_vis))
                poses_scene_vis[im_id] = poses

        for im_ind, (im_id, poses_img) in enumerate(poses_scene_vis.items()):

            # Retrieve camera intrinsics.
            if dataset == "hot3d":
                cam = pose_error_htt.create_camera_model(scene_camera[im_id])
            else:
                cam = scene_camera[im_id]["cam_K"]

            # Load the color and depth images and prepare images for rendering.
            rgb = None
            if args.vis_rgb:
                # rgb_tpath is an alias refering to the sensor|modality image paths on which the poses are rendered
                im_tpath = tpath_keys["rgb_tpath"]
                # check for BOP classic (itodd)
                rgb_available = dataset_params.sensor_has_modality(
                    dp_split, scene_sensor, "rgb"
                )
                if im_tpath == "rgb_tpath" and not rgb_available:
                    im_tpath = "gray_tpath"

                rgb = inout.load_im(
                    dp_split[im_tpath].format(scene_id=scene_id, im_id=im_id)
                )
                # if image is grayscale (e.g. quest3), convert it to 3 channels
                if rgb.ndim == 2:
                    rgb = np.dstack([rgb, rgb, rgb])
                else:
                    rgb = rgb[:, :, :3]  # should we keep this?

            depth = None
            if args.vis_depth_diff or (args.vis_rgb and args.vis_rgb_resolve_visib):
                depth_available = dataset_params.sensor_has_modality(
                    dp_split, scene_sensor, "depth"
                )
                if not depth_available:
                    misc.log(
                        f"{scene_sensor} has no depth data, skipping depth visualization"
                    )
                    args.vis_depth_diff = False
                    args.vis_rgb_resolve_visib = False
                else:
                    depth = inout.load_depth(
                        dp_split[tpath_keys["depth_tpath"]].format(
                            scene_id=scene_id, im_id=im_id
                        )
                    )
                    depth *= scene_camera[im_id]["depth_scale"]  # Convert to [mm].

            if args.mode == "gt":

                # Path to the output RGB visualization.
                split = (
                    "{}_{}".format(args.dataset_split, scene_sensor)
                    if scene_sensor
                    else args.dataset_split
                )
                vis_rgb_path = None
                if args.vis_rgb:
                    vis_rgb_path = args.vis_rgb_tpath.format(
                        vis_path=args.vis_path,
                        dataset=args.dataset,
                        split=split,
                        scene_id=scene_id,
                        im_id=im_id,
                    )

                # Path to the output depth difference visualization.
                vis_depth_diff_path = None
                if args.vis_depth_diff:
                    vis_depth_diff_path = args.vis_depth_diff_tpath.format(
                        vis_path=args.vis_path,
                        dataset=args.dataset,
                        split=split,
                        scene_id=scene_id,
                        im_id=im_id,
                    )
            else:
                # Visualization name.
                vis_name = "{im_id:06d}".format(im_id=im_id)
                # Path to the output RGB visualization.
                vis_rgb_path = None
                if args.vis_rgb:
                    vis_rgb_path = args.vis_rgb_tpath.format(
                        vis_path=args.vis_path,
                        result_name=result_name,
                        scene_id=scene_id,
                        vis_name=vis_name,
                    )

                # Path to the output depth difference visualization.
                vis_depth_diff_path = None
                if args.vis_depth_diff:
                    vis_depth_diff_path = args.vis_depth_diff_tpath.format(
                        vis_path=args.vis_path,
                        result_name=result_name,
                        scene_id=scene_id,
                        vis_name=vis_name,
                    )

            visualization.vis_object_poses(
                poses=poses_img,
                K=cam,
                renderer=ren,
                rgb=rgb,
                depth=depth,
                vis_rgb_path=vis_rgb_path,
                vis_depth_diff_path=vis_depth_diff_path,
                vis_rgb_resolve_visib=args.vis_rgb_resolve_visib,
            )
        #     break
        # break


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.vis_path is None:
        vis_type = "gt" if args.mode == "gt" else "est"
        args.vis_path = os.path.join(config.output_path, f"vis_{vis_type}_poses")
    main(args)
