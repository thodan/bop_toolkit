# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates performance scores for 6D object pose estimation tasks.

Errors of the pose estimates need to be pre-calculated with eval_calc_errors.py.

Currently supported tasks (see [1]):
- SiSo (a single instance of a single object)

For evaluation in the BOP paper [1], the following parameters were used:
 - n_top = 1
 - visib_gt_min = 0.1
 - error_type = 'vsd'
 - vsd_cost = 'step'
 - vsd_delta = 15
 - vsd_tau = 20
 - correct_th['vsd'] = 0.3

 [1] Hodan, Michel et al. BOP: Benchmark for 6D Object Pose Estimation, ECCV'18.
"""

import os
import time
import argparse

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import pose_matching
from bop_toolkit_lib import score

# Get the base name of the file without the .py extension
file_name = os.path.splitext(os.path.basename(__file__))[0]
logger = misc.get_logger(file_name)
# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
    # Threshold of correctness for different pose error functions.
    "correct_th": {
        "vsd": [0.3],
        "mssd": [0.2],
        "mspd": [10],
        "cus": [0.5],
        "rete": [5.0, 5.0],  # [deg, cm].
        "re": [5.0],  # [deg].
        "te": [5.0],  # [cm].
        "proj": [5.0],  # [px].
        "ad": [0.1],
        "add": [0.1],
        "adi": [0.1],
    },
    # Pose errors that will be normalized by object diameter before thresholding.
    "normalized_by_diameter": ["ad", "add", "adi", "mssd"],
    # Pose errors that will be normalized the image width before thresholding.
    "normalized_by_im_width": ["mspd"],
    # by default, we consider only objects that are at least 10% visible
    "visib_gt_min": -1,
    # Whether to use the visible surface fraction of a valid GT pose in the 6D detection
    "ignore_object_visible_less_than_visib_gt_min": True,
    # Paths (relative to p['eval_path']) to folders with pose errors calculated
    # using eval_calc_errors.py.
    # Example: 'hodan-iros15_lm-test/error=vsd_ntop=1_delta=15_tau=20_cost=step'
    "error_dir_paths": [
        r"/path/to/calculated/errors",
    ],
    # Folder for the calculated pose errors and performance scores.
    "eval_path": config.eval_path,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # File with a list of estimation targets to consider. The file is assumed to
    # be stored in the dataset folder.
    "targets_filename": "test_targets_bop19.json",
    # Template of path to the input file with calculated errors.
    "error_tpath": os.path.join(
        "{eval_path}", "{error_dir_path}", "errors_{scene_id:06d}.json"
    ),
    # Template of path to the output file with established matches and calculated
    # scores.
    "out_matches_tpath": os.path.join(
        "{eval_path}", "{error_dir_path}", "matches_{score_sign}.json"
    ),
    "out_scores_tpath": os.path.join(
        "{eval_path}", "{error_dir_path}", "scores_{score_sign}.json"
    ),
    "eval_mode": "localization",  # Options: 'localization', 'detection'.
    "eval_modality": None,  # Options: depends on the dataset, e.g. for hot3d 'rgb'
    "max_num_estimates_per_image": 100,  # Maximum number of estimates per image. Only used for detection tasks.
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Define the command line arguments.
for err_type in p["correct_th"]:
    parser.add_argument(
        "--correct_th_" + err_type,
        default=",".join(map(str, p["correct_th"][err_type])),
    )

parser.add_argument(
    "--normalized_by_diameter", default=",".join(p["normalized_by_diameter"])
)
parser.add_argument(
    "--normalized_by_im_width", default=",".join(p["normalized_by_im_width"])
)
parser.add_argument("--visib_gt_min", default=p["visib_gt_min"])
parser.add_argument(
    "--ignore_object_visible_less_than_visib_gt_min",
    action="store_true",
    default=p["ignore_object_visible_less_than_visib_gt_min"],
)
parser.add_argument(
    "--error_dir_paths",
    default=",".join(p["error_dir_paths"]),
    help="Comma-sep. paths to errors from eval_calc_errors.py.",
)
parser.add_argument("--eval_path", default=p["eval_path"])
parser.add_argument("--datasets_path", default=p["datasets_path"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
parser.add_argument("--error_tpath", default=p["error_tpath"])
parser.add_argument("--out_matches_tpath", default=p["out_matches_tpath"])
parser.add_argument("--out_scores_tpath", default=p["out_scores_tpath"])
parser.add_argument("--eval_mode", default=p["eval_mode"])
# Process the command line arguments.
args = parser.parse_args()

for err_type in p["correct_th"]:
    p["correct_th"][err_type] = list(
        map(float, args.__dict__["correct_th_" + err_type].split(","))
    )

p["normalized_by_diameter"] = args.normalized_by_diameter.split(",")
p["normalized_by_im_width"] = args.normalized_by_im_width.split(",")
p["visib_gt_min"] = float(args.visib_gt_min)
p["error_dir_paths"] = args.error_dir_paths.split(",")
p["eval_path"] = str(args.eval_path)
p["datasets_path"] = str(args.datasets_path)
p["targets_filename"] = str(args.targets_filename)
p["error_tpath"] = str(args.error_tpath)
p["out_matches_tpath"] = str(args.out_matches_tpath)
p["out_scores_tpath"] = str(args.out_scores_tpath)
p["eval_mode"] = str(args.eval_mode)
p["ignore_object_visible_less_than_visib_gt_min"] = bool(
    args.ignore_object_visible_less_than_visib_gt_min
)

logger.info("-----------")
logger.info("Parameters:")
for k, v in p.items():
    logger.info("- {}: {}".format(k, v))
logger.info("-----------")


# Calculation of the performance scores.
# ------------------------------------------------------------------------------
for error_dir_path in p["error_dir_paths"]:
    logger.info("Processing: {}".format(error_dir_path))

    time_start = time.time()

    # Parse info about the errors from the folder name.
    error_sign = os.path.basename(error_dir_path)
    err_type = str(error_sign.split("_")[0].split("=")[1])
    n_top = int(error_sign.split("_")[1].split("=")[1])
    result_info = os.path.basename(os.path.dirname(error_dir_path)).split("_")
    method = result_info[0]
    dataset_info = result_info[1].split("-")
    dataset = dataset_info[0]
    split = dataset_info[1]
    split_type = dataset_info[2] if len(dataset_info) > 2 else None

    # Evaluation signature.
    score_sign = misc.get_score_signature(p["correct_th"][err_type], p["visib_gt_min"])

    if dataset == "xyzibd":
        p["max_num_estimates_per_image"] = 200

    logger.info(
        "Calculating score - error: {}, method: {}, dataset: {}.".format(
            err_type, method, dataset
        )
    )

    # Load dataset parameters.
    dp_split = dataset_params.get_split_params(
        p["datasets_path"], dataset, split, split_type
    )

    model_type = "eval"
    dp_model = dataset_params.get_model_params(p["datasets_path"], dataset, model_type)

    # Load info about the object models.
    models_info = inout.load_json(dp_model["models_info_path"], keys_to_int=True)

    # Load the estimation targets to consider.
    targets = inout.load_json(
        os.path.join(dp_split["base_path"], p["targets_filename"]))

    # Organize the targets by scene, image and object.
    logger.info("Organizing estimation targets...")
    # targets_org : {"scene_id": {"im_id": {5: {"im_id": 3, "inst_count": 1, "obj_id": 3, "scene_id": 48}}}}
    targets_org = {}
    for target in targets:
        if p["eval_mode"] == "localization":
            assert "inst_count" in target, "inst_count is required for localization mode" 
            targets_org.setdefault(target["scene_id"], {}).setdefault(target["im_id"], {})[target["obj_id"]] = target
        else:
            targets_org.setdefault(target["scene_id"], {})[target["im_id"]] = target

    # Go through the test scenes and match estimated poses to GT poses.
    # ----------------------------------------------------------------------------
    estimates = []
    matches = []  # Stores info about the matching pose estimate for each GT pose.
    scene_im_widths = {}
    for scene_id, scene_targets in targets_org.items():
        logger.info("Processing scene {} of {}...".format(scene_id, dataset))

        tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)
        scene_modality = dataset_params.get_scene_sensor_or_modality(dp_split["eval_modality"], scene_id)
        scene_sensor = dataset_params.get_scene_sensor_or_modality(dp_split["eval_sensor"], scene_id)

        # Load GT poses for the current scene.
        scene_gt = inout.load_scene_gt(
            dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id)
        )
        # Load info about the GT poses (e.g. visibility) for the current scene.
        scene_gt_info = inout.load_json(
            dp_split[tpath_keys["scene_gt_info_tpath"]].format(scene_id=scene_id), keys_to_int=True
        )
        # Load ground truth camera 
        scene_camera = inout.load_scene_camera(dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id))

        # Handle change of image size location between BOP19 and BOP24 dataset formats
        scene_im_widths[scene_id] = dataset_params.get_im_size(dp_split, scene_modality, scene_sensor)[0]

        # Keep GT poses only for the selected targets.
        scene_gt_curr = {}
        scene_gt_curr_info = {}
        scene_gt_valid = {}
        for im_id, im_targets in scene_targets.items():

            # Create im_targets directly from scene_gt and scene_gt_info.
            im_gt = scene_gt[im_id]
            im_gt_info = scene_gt_info[im_id]
            # We need to re-define the target file for 6D detection tasks because:
            # We want to consider all GT, not only GT>visib_gt_min since we want to ignore estimation matches with GT < visib_gt_min.  
            if p["eval_mode"] == "detection":
                im_targets = inout.get_im_targets(im_gt=im_gt, im_gt_info=im_gt_info, visib_gt_min=p["visib_gt_min"], eval_mode=p["eval_mode"])

            scene_gt_curr[im_id] = scene_gt[im_id]
            scene_gt_curr_info[im_id] = scene_gt_info[im_id]
            # Determine which GT poses are valid.
            im_gt = scene_gt[im_id]
            im_gt_info = scene_gt_info[im_id]
            scene_gt_valid[im_id] = [True] * len(im_gt)
            # For 6D detection, we consider all GT are valid, scene_gt_valid = True
            # For 6D localization, a GT is valid when it's in target_file and its visiblity > visib_gt_min
            if p["eval_mode"] == "localization":
                if p["visib_gt_min"] >= 0:
                    for gt_id, gt in enumerate(im_gt):
                        is_target = gt["obj_id"] in im_targets.keys()
                        is_visib = im_gt_info[gt_id]["visib_fract"] >= p["visib_gt_min"]
                        scene_gt_valid[im_id][gt_id] = is_target and is_visib
                else:
                    # k most visible GT poses are considered valid, where k is given by
                    # the "inst_count" item loaded from "targets_filename".
                    gt_ids_sorted = sorted(
                        range(len(im_gt)),
                        key=lambda gt_id: im_gt_info[gt_id]["visib_fract"],
                        reverse=True,
                    )
                    to_add = {
                        obj_id: trg["inst_count"] for obj_id, trg in im_targets.items()
                    }
                    for gt_id in gt_ids_sorted:
                        obj_id = im_gt[gt_id]["obj_id"]
                        if obj_id in to_add.keys() and to_add[obj_id] > 0:
                            scene_gt_valid[im_id][gt_id] = True
                            to_add[obj_id] -= 1
                        else:
                            scene_gt_valid[im_id][gt_id] = False

        # Load pre-calculated errors of the pose estimates w.r.t. the GT poses.
        scene_errs_path = p["error_tpath"].format(
            eval_path=p["eval_path"], error_dir_path=error_dir_path, scene_id=scene_id
        )

        scene_errs = inout.load_json(scene_errs_path, keys_to_int=True)

        # Normalize the errors by the object diameter.
        if err_type in p["normalized_by_diameter"]:
            for err in scene_errs:
                diameter = float(models_info[err["obj_id"]]["diameter"])
                for gt_id in err["errors"].keys():
                    err["errors"][gt_id] = [e / diameter for e in err["errors"][gt_id]]

        # Normalize the errors by the image width.
        if err_type in p["normalized_by_im_width"]:
            for err in scene_errs:
                factor = 640.0 / scene_im_widths[err["scene_id"]]
                for gt_id in err["errors"].keys():
                    err["errors"][gt_id] = [factor * e for e in err["errors"][gt_id]]

        # Match the estimated poses to the ground-truth poses.
        matches += pose_matching.match_poses_scene(
            scene_id,
            scene_gt_curr,
            scene_gt_curr_info,
            scene_gt_valid,
            scene_errs,
            p["correct_th"][err_type],
            n_top,
        )

        # Keep all the estimates
        estimates += scene_errs

    # Calculate the performance scores.
    # ----------------------------------------------------------------------------
    # 6D object localization scores (SiSo if n_top = 1).
    if p["eval_mode"] == "localization":
        scores = score.calc_localization_scores(
            dp_split["scene_ids"], dp_model["obj_ids"], matches, n_top
        )
    elif p["eval_mode"] == "detection":
        scores = score.calc_pose_detection_scores(
            dp_split["scene_ids"],
            dp_model["obj_ids"],
            matches,
            estimates,
            visib_gt_min=p["visib_gt_min"],
        )
    else:
        raise ValueError("Unknown eval_mode: {}".format(p["eval_mode"]))
    # Save scores.
    scores_path = p["out_scores_tpath"].format(
        eval_path=p["eval_path"], error_dir_path=error_dir_path, score_sign=score_sign
    )
    inout.save_json(scores_path, scores)

    # Save matches.
    matches_path = p["out_matches_tpath"].format(
        eval_path=p["eval_path"], error_dir_path=error_dir_path, score_sign=score_sign
    )
    inout.save_json(matches_path, matches)

    time_total = time.time() - time_start
    logger.info("Matching and score calculation took {}s.".format(time_total))

logger.info("Done.")
