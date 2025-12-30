import argparse
import os

from bop_toolkit_lib import config, misc

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
        # Whether to use the original model color.
        "vis_orig_color": False,
        # Type of the renderer (used for the VSD pose error function).
        # Options: 'vispy', 'cpp', 'python'. 'htt' is mandatory for "hot3d" dataset.
        "renderer_type": "vispy",
        # Folder containing the BOP datasets.
        "datasets_path": config.datasets_path,
        # Folder for output visualisations. The default value depends on script's mode
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
        #########
        # Which sensor to visualize. By default it uses the evaluation modality set
        # in dataset_params.py. Set to None for rendering PBR images or BOP core datasets.
        # Set to sensor for new BOP core sets, e.g. "photoneo".
        #########
        # Modality used to visualize ground truth, default to eval modality.
        "modality": None,
        # Sensor used to visualize ground truth, default to eval sensor.
        "sensor": None,
        # Select ID's of scenes, images and GT poses to be processed, otherwise use all of them
        "scene_ids": None,
        "im_ids": None,
        "gt_ids": None,
        # Path templates for output images.
        "vis_path_template": os.path.join(
            "{vis_path}", "{dataset}", "{split}", "{scene_id:06d}", "{im_id:06d}{suffix}.jpg"
        ),
    },
    "est": {
        # Top N pose estimates (with the highest score) to be visualized for each
        # object in each image. 0 = all estimates, -1 = given by the number of GT poses.
        "n_top": 0,
        # Name of a file with pose estimates to visualize (assumed to be stored in
        # folder config.eval_path). See docs/bop_challenge_2019.md for a description
        # of the format. Example results can be found at:
        # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip
        "result_filename": None,
        # Folder with results to be evaluated.
        "results_path": config.results_path,
        "vis_path_template": os.path.join(
            "{vis_path}", "{result_name}", "{scene_id:06d}", "{im_id:06d}{suffix}.jpg"
        )
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
        default=c_defs["renderer_type"],
        help="Renderer type (vispy, cpp, python, htt)",
    )
    common_parser.add_argument(
        "--datasets_path",
        default=c_defs["datasets_path"],
        help="Path to BOP datasets",
    )
    common_parser.add_argument(
        "--vis_path",
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
    parser_gt.add_argument("--dataset", default=gt_defs["dataset"])
    parser_gt.add_argument("--dataset_split", default=gt_defs["dataset_split"])
    parser_gt.add_argument(
        "--dataset_split_type", default=gt_defs["dataset_split_type"]
    )
    parser_gt.add_argument(
        "--targets_filename",
        default=gt_defs["targets_filename"],
        help="JSON file with targets (scene/im_ids) to visualize",
    )
    parser_gt.add_argument("--modality", default=gt_defs["modality"])
    parser_gt.add_argument("--sensor", default=gt_defs["sensor"])

    parser_gt.add_argument("--scene_ids", help="Comma-separated scene IDs")
    parser_gt.add_argument("--im_ids", help="Comma-separated image IDs")
    parser_gt.add_argument("--gt_ids", help="Comma-separated GT object IDs")

    parser_gt.add_argument(
        "--vis_rgb_tpath",
        default=gt_defs["vis_rgb_tpath"],
        help="Template path for output RGB images",
    )
    parser_gt.add_argument(
        "--vis_depth_diff_tpath",
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
        default=est_defs["result_filename"],
        required=True,
        help="Result file",
    )
    parser_est.add_argument(
        "--results_path",
        default=est_defs["results_path"],
        help="Path to results folder",
    )
    parser_est.add_argument(
        "--vis_rgb_tpath",
        default=est_defs["vis_rgb_tpath"],
        help="Template path for output RGB images",
    )
    parser_est.add_argument(
        "--vis_depth_diff_tpath",
        default=est_defs["vis_depth_diff_tpath"],
        help="Template path for output depth difference images",
    )
    return parser


def postprocess_args(args):

    if args.mode == "gt":

        if args.modality == "depth":
            raise ValueError("Modality for GT visualization cannot be 'depth'")

        for attr in ["scene_ids", "im_ids", "gt_ids"]:
            val = getattr(args, attr)
            if val is not None:
                setattr(args, attr, [int(x) for x in val.split(",")])
                setattr(args, attr, [int(x) for x in val.split(",")])

    return args