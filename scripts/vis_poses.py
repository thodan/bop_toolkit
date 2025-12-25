"""
Visualizes object models in the GT/estimated poses.
The script visualize datasets in the classical BOP19 format as well as the HOT3D dataset in H3 BOP24 format.
"""

import argparse
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
        # True = one visualization for each (im_id, obj_id), False = one per im_id.
        "vis_per_obj_id": False,
        # Names of files with pose estimates to visualize (assumed to be stored in
        # folder config.eval_path).
        "result_filenames": [],
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
    misc.add_argument_bool(parser_est, "vis_per_obj_id", est_defs["vis_per_obj_id"])
    parser_est.add_argument(
        "--result_filenames",
        type=str,
        default=",".join(est_defs["result_filenames"]),
        help="Comma-separated names of result files",
    )
    parser_est.add_argument(
        "--results_path",
        type=str,
        default=est_defs["results_path"],
        help="Path to results folder",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.vis_path is None:
        vis_type = "gt" if args.mode == "gt" else "est"
        args.vis_path = os.path.join(config.output_path, f"vis_{vis_type}_poses")
