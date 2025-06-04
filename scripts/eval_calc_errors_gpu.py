# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates error of 6D object pose estimates."""

import os
import time
import argparse
import copy
import numpy as np


from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import pose_error_gpu

# Get the base name of the file without the .py extension
file_name = os.path.splitext(os.path.basename(__file__))[0]
logger = misc.get_logger(file_name)

try:
    import torch
except ImportError as e:
    logger.error('torch is required for gpu based evaluation (see bop_toolkit/README.md)')
    raise e

# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
    # Top N pose estimates (with the highest score) to be evaluated for each
    # object class in each image.
    # Options: 0 = all, -1 = given by the number of GT poses.
    "n_top": 1,
    # by default, we consider only objects that are at least 10% visible
    "visib_gt_min": -1,
    # Pose error function.
    # Options: 'vsd', 'mssd', 'mspd', 'ad', 'adi', 'add', 'cus', 're', 'te, etc.
    "error_type": "vsd",
    # VSD parameters.
    "vsd_deltas": {
        "hb": 15,
        "icbin": 15,
        "icmi": 15,
        "itodd": 5,
        "lm": 15,
        "lmo": 15,
        "ruapc": 15,
        "tless": 15,
        "tudl": 15,
        "tyol": 15,
        "ycbv": 15,
        "hope": 15,
    },
    "vsd_taus": list(np.arange(0.05, 0.51, 0.05)),
    "vsd_normalized_by_diameter": True,
    # MSSD/MSPD parameters (see misc.get_symmetry_transformations).
    "max_sym_disc_step": 0.01,
    # Whether to ignore/break if some errors are missing.
    "skip_missing": True,
    # Type of the renderer (used for the VSD pose error function).
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    # Names of files with results for which to calculate the errors (assumed to be
    # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
    # description of the format. Example results can be found at:
    # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip
    "result_filenames": [
        "/path/to/csv/with/results",
    ],
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder for the calculated pose errors and performance scores.
    "eval_path": config.eval_path,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # File with a list of estimation targets to consider. The file is assumed to
    # be stored in the dataset folder.
    "targets_filename": "test_targets_bop19.json",
    # Template of path to the output file with calculated errors.
    "out_errors_tpath": os.path.join(
        "{eval_path}", "{result_name}", "{error_sign}", "errors_{scene_id:06d}.json"
    ),
    "num_workers": config.num_workers,  # Number of parallel workers for the calculation of errors.
    "eval_mode": "localization",  # Options: 'localization', 'detection'.
    "max_num_estimates_per_image": 100,  # Maximum number of estimates per image. Only used for detection tasks.
    "device": "cuda:0",  # use "device" for torch computations. If not available, falls back to "cpu".
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
vsd_deltas_str = ",".join(["{}:{}".format(k, v) for k, v in p["vsd_deltas"].items()])

parser = argparse.ArgumentParser()
parser.add_argument("--n_top", default=p["n_top"])
parser.add_argument("--visib_gt_min", default=p["visib_gt_min"])
parser.add_argument("--error_type", default=p["error_type"])
parser.add_argument("--vsd_deltas", default=vsd_deltas_str)
parser.add_argument("--vsd_taus", default=",".join(map(str, p["vsd_taus"])))
parser.add_argument(
    "--vsd_normalized_by_diameter", default=p["vsd_normalized_by_diameter"]
)
parser.add_argument("--max_sym_disc_step", default=p["max_sym_disc_step"])
parser.add_argument("--skip_missing", default=p["skip_missing"])
parser.add_argument("--renderer_type", default=p["renderer_type"])
parser.add_argument(
    "--result_filenames",
    default=",".join(p["result_filenames"]),
    help="Comma-separated names of files with results.",
)
parser.add_argument("--results_path", default=p["results_path"])
parser.add_argument("--eval_path", default=p["eval_path"])
parser.add_argument("--datasets_path", default=p["datasets_path"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
parser.add_argument("--out_errors_tpath", default=p["out_errors_tpath"])
parser.add_argument("--num_workers", default=p["num_workers"])
parser.add_argument("--eval_mode", default=p["eval_mode"])
parser.add_argument("--device", type=str, default=p["device"])
args = parser.parse_args()

p["n_top"] = int(args.n_top)
p["visib_gt_min"] = float(args.visib_gt_min)
p["error_type"] = str(args.error_type)
assert p["error_type"] in [
    "mssd",
    "mspd",
], "Only mssd and mspd are supported!"
p["vsd_deltas"] = {
    str(e.split(":")[0]): float(e.split(":")[1]) for e in args.vsd_deltas.split(",")
}
p["vsd_taus"] = list(map(float, args.vsd_taus.split(",")))
p["vsd_normalized_by_diameter"] = bool(args.vsd_normalized_by_diameter)
p["max_sym_disc_step"] = float(args.max_sym_disc_step)
p["skip_missing"] = bool(args.skip_missing)
p["renderer_type"] = str(args.renderer_type)
p["result_filenames"] = args.result_filenames.split(",")
p["results_path"] = str(args.results_path)
p["eval_path"] = str(args.eval_path)
p["datasets_path"] = str(args.datasets_path)
p["targets_filename"] = str(args.targets_filename)
p["out_errors_tpath"] = str(args.out_errors_tpath)
p["num_workers"] = int(args.num_workers)
p["eval_mode"] = str(args.eval_mode)


def is_specific_device_available(device_str):
    if not torch.cuda.is_available():
        return False  # CUDA is not available at all

    # Handle "cuda" (defaults to cuda:0)
    if device_str == "cuda":
        return torch.cuda.device_count() >= 1

    # Handle "cuda:X"
    if device_str.startswith("cuda:"):
        try:
            device_id = int(device_str.split(":")[1])
            return device_id < torch.cuda.device_count()
        except (ValueError, IndexError):
            return False  # Invalid format

    return False  # Not a CUDA device string


if is_specific_device_available(args.device):
    device = torch.device(args.device)
else:
    logger.error(f"###########################################")
    logger.error(f"CUDA device {args.device} is not available!")
    logger.error(f"Falling back to 'cpu'")
    logger.error(f"###########################################\n")
    device = torch.device("cpu")

logger.info("-----------")
logger.info(f"Using device={device}")
logger.info("Parameters:")
for k, v in p.items():
    logger.info("- {}: {}".format(k, v))
logger.info("-----------")

# Error calculation.
# ------------------------------------------------------------------------------
for result_filename in p["result_filenames"]:
    logger.info("Processing: {}".format(result_filename))

    ests_counter = 0
    time_start = time.time()

    # Parse info about the method and the dataset from the filename.
    result_name, method, dataset, split, split_type, _ = inout.parse_result_filename(result_filename)
    split_type_str = " - " + split_type if split_type is not None else ""

    # Load dataset parameters.
    dp_split = dataset_params.get_split_params(
        p["datasets_path"], dataset, split, split_type
    )

    if dataset == "xyzibd":
        p["max_num_estimates_per_image"] = 200

    model_type = "eval"
    dp_model = dataset_params.get_model_params(p["datasets_path"], dataset, model_type)

    # Load object models.
    models = {}
    if p["error_type"] in ["ad", "add", "adi", "mssd", "mspd", "proj"]:
        logger.info("Loading object models...")
        for obj_id in dp_model["obj_ids"]:
            obj_pcd = inout.load_ply(dp_model["model_tpath"].format(obj_id=obj_id))[
                "pts"
            ]
            logger.info(f" Loaded object {obj_id}: {obj_pcd.shape[0]} points")
            models[obj_id] = torch.from_numpy(obj_pcd).float().to(device)

    # Load models info.
    models_info = None
    if p["error_type"] in ["ad", "add", "adi", "vsd", "mssd", "mspd", "cus"]:
        models_info = inout.load_json(dp_model["models_info_path"], keys_to_int=True)

    # Get sets of symmetry transformations for the object models.
    models_sym = None
    if p["error_type"] in ["mssd", "mspd"]:
        models_sym = {}
        for obj_id in dp_model["obj_ids"]:
            obj_sym = misc.get_symmetry_transformations(
                models_info[obj_id], p["max_sym_disc_step"]
            )
            for sym in obj_sym:
                sym["R"] = torch.from_numpy(sym["R"]).float().to(device)
                sym["t"] = torch.from_numpy(sym["t"]).float().to(device)
            logger.info(f" Object {obj_id} has {len(obj_sym)-1} symmetries")
            models_sym[obj_id] = obj_sym

    # Load the estimation targets.
    targets = inout.load_json(
        os.path.join(dp_split["base_path"], p["targets_filename"])
    )

    # Organize the targets by scene, image and object.
    logger.info("Organizing estimation targets...")
    targets_org = {}
    for target in targets:
        if p["eval_mode"] == "localization":
            assert "inst_count" in target, "inst_count is required for localization mode" 
            targets_org.setdefault(target["scene_id"], {}).setdefault(target["im_id"], {})[target["obj_id"]] = target
        else:
            targets_org.setdefault(target["scene_id"], {})[target["im_id"]] = target

    # Load pose estimates.
    logger.info("Loading pose estimates...")
    max_num_estimates_per_image = p["max_num_estimates_per_image"] if p["eval_mode"] == "detection" else None
    result_path = os.path.join(p["results_path"], result_filename)
    ests = inout.load_bop_results(result_path, max_num_estimates_per_image=max_num_estimates_per_image)

    # Organize the pose estimates by scene, image and object.
    logger.info("Organizing pose estimates...")
    ests_org = {}
    for est in ests:
        ests_org.setdefault(est["scene_id"], {}).setdefault(
            est["im_id"], {}
        ).setdefault(est["obj_id"], []).append(est)

    # create estimate templates
    estimate_templates = {}
    for obj_id in dp_model["obj_ids"]:
        estimate_templates[obj_id] = {
            "R_e": [],
            "t_e": [],
            "R_g": [],
            "t_g": [],
            "cam_K": [],
            "im_id": [],
            "est_id": [],
            "gt_id": [],
            "scores": [],
            "gt_visib_fract": [],
        }
    all_estimates = copy.deepcopy(estimate_templates)

    for scene_id, scene_targets in targets_org.items():
        # for each scene, organize the estimates per object as each object
        est_per_object = copy.deepcopy(estimate_templates)

        tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)

        # Load camera and GT poses for the current scene.
        scene_camera = inout.load_scene_camera(
            dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id)
        )
        scene_gt = inout.load_scene_gt(
            dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id)
        )
        scene_gt_info = inout.load_scene_gt(
            dp_split[tpath_keys["scene_gt_info_tpath"]].format(scene_id=scene_id)
        )

        for im_ind, (im_id, im_targets) in enumerate(scene_targets.items()):

            if im_ind % 10 == 0:
                logger.info(f"Processing scene: {scene_id}, im: {im_ind}")

            # Create im_targets directly from scene_gt and scene_gt_info.
            im_gt = scene_gt[im_id]
            im_gt_info = scene_gt_info[im_id]
            if p["eval_mode"] == "detection":
                im_targets = inout.get_im_targets(im_gt=im_gt, im_gt_info=im_gt_info, visib_gt_min=p["visib_gt_min"], eval_mode=p["eval_mode"])

            for obj_id, target in im_targets.items():

                # The required number of top estimated poses.
                if p["n_top"] == 0:  # All estimates are considered.
                    n_top_curr = None
                elif p["n_top"] == -1:  # Given by the number of GT poses.
                    # n_top_curr = sum([gt['obj_id'] == obj_id for gt in scene_gt[im_id]])
                    n_top_curr = target["inst_count"]
                else:
                    n_top_curr = p["n_top"]

                # Get the estimates.
                try:
                    obj_ests = ests_org[scene_id][im_id][obj_id]
                    obj_count = len(obj_ests)
                except KeyError:
                    obj_ests = []
                    obj_count = 0

                # Check the number of estimates.
                if not p["skip_missing"] and obj_count < n_top_curr:
                    raise ValueError(
                        "Not enough estimates for scene: {}, im: {}, obj: {} "
                        "(provided: {}, expected: {})".format(
                            scene_id, im_id, obj_id, obj_count, n_top_curr
                        )
                    )

                # Sort the estimates by score (in descending order).
                obj_ests_sorted = sorted(
                    enumerate(obj_ests), key=lambda x: x[1]["score"], reverse=True
                )

                # Select the required number of top estimated poses.
                obj_ests_sorted = obj_ests_sorted[slice(0, n_top_curr)]
                ests_counter += len(obj_ests_sorted)

                # Calculate error of each pose estimate w.r.t. all GT poses of the same
                # object class.
                for est_id, est in obj_ests_sorted:
                    for gt_id, gt in enumerate(scene_gt[im_id]):
                        if gt["obj_id"] != obj_id:
                            continue

                        # Ground-truth pose.
                        est_per_object[obj_id]["R_g"].append(gt["cam_R_m2c"])
                        est_per_object[obj_id]["t_g"].append(gt["cam_t_m2c"])
                        est_per_object[obj_id]["gt_visib_fract"].append(
                            scene_gt_info[im_id][gt_id]["visib_fract"]
                        )
                        # BOP24 datasets do not have "cam_K" keys but "cam_model" 
                        # TODO: handle the case of H3 dataset which are pinhole but have "cam_model" key (like HANDAL)
                        if "cam_K" in scene_camera[im_id]:
                            est_per_object[obj_id]["cam_K"].append(
                                scene_camera[im_id]["cam_K"]
                            )

                        # Estimated pose.
                        est_per_object[obj_id]["R_e"].append(est["R"])
                        est_per_object[obj_id]["t_e"].append(est["t"])

                        # Image ID, object ID and score.
                        est_per_object[obj_id]["im_id"].append(im_id)
                        est_per_object[obj_id]["est_id"].append(est_id)
                        est_per_object[obj_id]["gt_id"].append(gt_id)
                        est_per_object[obj_id]["scores"].append(est["score"])

        # calculate errors
        scene_errs = {}
        for obj_id in est_per_object:
            if len(est_per_object[obj_id]["R_e"]) == 0:
                continue
            for name in est_per_object[obj_id]:
                est_per_object[obj_id][name] = (
                    torch.from_numpy(np.asarray(est_per_object[obj_id][name]))
                    .float()
                    .to(device)
                )

            if p["error_type"] == "mssd":
                errors = pose_error_gpu.mssd_by_batch(
                    est_per_object[obj_id]["R_e"],
                    est_per_object[obj_id]["t_e"],
                    est_per_object[obj_id]["R_g"],
                    est_per_object[obj_id]["t_g"],
                    models[obj_id],
                    models_sym[obj_id],
                )
            elif p["error_type"] == "mspd":
                errors = pose_error_gpu.mspd_by_batch(
                    est_per_object[obj_id]["R_e"],
                    est_per_object[obj_id]["t_e"],
                    est_per_object[obj_id]["R_g"],
                    est_per_object[obj_id]["t_g"],
                    est_per_object[obj_id]["cam_K"],
                    models[obj_id],
                    models_sym[obj_id],
                )
            else:
                raise ValueError(f"Unknown error type: {p['error_type']}")

            errors = errors.cpu().numpy().tolist()
            for i in range(len(errors)):
                # for each scene_id, im_id, obj_id, est_id, save the errors wrt gt_id
                im_id = int(est_per_object[obj_id]["im_id"][i].item())
                est_id = int(est_per_object[obj_id]["est_id"][i].item())
                gt_id = int(est_per_object[obj_id]["gt_id"][i].item())
                score = est_per_object[obj_id]["scores"][i].item()
                gt_visib_fract = est_per_object[obj_id]["gt_visib_fract"][i].item()

                key_name = f"scene{scene_id:06d}_im_id{im_id:06d}obj_id{obj_id:06d}est_id{est_id:06d}"
                if key_name not in scene_errs:
                    scene_errs[key_name] = {
                        "scene_id": scene_id,
                        "im_id": im_id,
                        "obj_id": obj_id,
                        "est_id": est_id,
                        "score": score,
                        "gt_visib_fracts": {},
                        "errors": {},
                    }
                scene_errs[key_name]["errors"][gt_id] = [errors[i]]
                scene_errs[key_name]["gt_visib_fracts"][gt_id] = [gt_visib_fract]

        scene_errs = [v for k, v in scene_errs.items()]
        del est_per_object

        def save_errors(_error_sign, _scene_errs):
            # Save the calculated errors to a JSON file.
            errors_path = p["out_errors_tpath"].format(
                eval_path=p["eval_path"],
                result_name=result_name,
                error_sign=_error_sign,
                scene_id=scene_id,
            )
            misc.ensure_dir(os.path.dirname(errors_path))
            logger.info("Saving errors to: {}".format(errors_path))
            inout.save_json(errors_path, _scene_errs)

        # Save the calculated errors.
        error_sign = misc.get_error_signature(p["error_type"], p["n_top"])
        save_errors(error_sign, scene_errs)

    time_total = time.time() - time_start
    logger.info(
        "Calculation of errors for {} estimates took {}s (torch version).".format(
            ests_counter, time_total
        )
    )

logger.info("Done.")
