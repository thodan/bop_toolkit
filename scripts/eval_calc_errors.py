# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates error of 6D object pose estimates."""

import os
import time
import argparse
import copy
import numpy as np
import multiprocessing

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import pose_error

from bop_toolkit_lib import renderer
from bop_toolkit_lib import renderer_batch

# Get the base name of the file without the .py extension
file_name = os.path.splitext(os.path.basename(__file__))[0]
logger = misc.get_logger(file_name)

htt_available = False
try:
    from bop_toolkit_lib import pose_error_htt
    htt_available = True
except ImportError as e:
    logger.warn("""Missing hand_tracking_toolkit dependency, 
                mandatory if you are running evaluation on HOT3d. 
                Refer to the README.md for installation instructions.
                """)

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
args = parser.parse_args()

p["n_top"] = int(args.n_top)
p["visib_gt_min"] = float(args.visib_gt_min)
p["error_type"] = str(args.error_type)
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

logger.info("-----------")
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

    if p["error_type"] not in dp_split["supported_error_types"]:
        raise ValueError("""{} error is not among {} """
                         """supported error types: {}""".format(p["error_type"], dataset, dp_split["supported_error_types"]))
    
    model_type = "eval"
    dp_model = dataset_params.get_model_params(p["datasets_path"], dataset, model_type)

    # Load object models.
    models = {}
    if p["error_type"] in ["ad", "add", "adi", "mssd", "mspd", "proj"]:
        logger.info("Loading object models...")
        for obj_id in dp_model["obj_ids"]:
            models[obj_id] = inout.load_ply(
                dp_model["model_tpath"].format(obj_id=obj_id)
            )

    # Load models info.
    models_info = None
    if p["error_type"] in ["ad", "add", "adi", "vsd", "mssd", "mspd", "cus"]:
        models_info = inout.load_json(dp_model["models_info_path"], keys_to_int=True)

    # Get sets of symmetry transformations for the object models.
    models_sym = None
    if p["error_type"] in ["mssd", "mspd"]:
        models_sym = {}
        for obj_id in dp_model["obj_ids"]:
            models_sym[obj_id] = misc.get_symmetry_transformations(
                models_info[obj_id], p["max_sym_disc_step"]
            )

    # Initialize a renderer.
    ren = None
    if p["error_type"] in ["vsd", "cus"]:
        logger.info("Initializing renderer...")
        width, height = dp_split["im_size"]
        if p["num_workers"] == 1:
            ren = renderer.create_renderer(
                width, height, p["renderer_type"], mode="depth"
            )
        else:
            ren = renderer_batch.BatchRenderer(
                width,
                height,
                p["renderer_type"],
                mode="depth",
                num_workers=p["num_workers"],
                tmp_dir=os.path.join(
                    p["results_path"], p["eval_path"], f"tmp{int(time.time())}"
                ),
            )
        for obj_id in dp_model["obj_ids"]:
            ren.add_object(obj_id, dp_model["model_tpath"].format(obj_id=obj_id))

    # Load the estimation targets.
    targets = inout.load_json(
        os.path.join(dp_split["base_path"], p["targets_filename"])
    )

    # Organize the targets by scene, image and object.
    logger.info("Organizing estimation targets...")
    # targets_org : {"scene_id": {"im_id": {5: {"im_id": 3, "inst_count": 1, "obj_id": 3, "scene_id": 48}}}}
    targets_org = {}
    
    for target in targets:
        if p["eval_mode"] == "localization":
            assert "inst_count" in target, "inst_count is required for localization eval_mode" 
            targets_org.setdefault(target["scene_id"], {}).setdefault(target["im_id"], {})[target["obj_id"]] = target
        else:
            targets_org.setdefault(target["scene_id"], {})[target["im_id"]] = target

    # Load pose estimates.
    logger.info("Loading pose estimates...")
    max_num_estimates_per_image = p["max_num_estimates_per_image"] if p["eval_mode"] == "detection" else None
    ests = inout.load_bop_results(os.path.join(p["results_path"], result_filename), max_num_estimates_per_image=max_num_estimates_per_image)

    # Organize the pose estimates by scene, image and object.
    logger.info("Organizing pose estimates...")
    ests_org = {}
    for est in ests:
        ests_org.setdefault(est["scene_id"], {}).setdefault(
            est["im_id"], {}
        ).setdefault(est["obj_id"], []).append(est)

    for scene_id, scene_targets in targets_org.items():
        logger.info("Processing scene {} of {}...".format(scene_id, dataset))
        tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)

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

        # collect all the images and their targets
        im_meta_datas = []
        
        for im_ind, (im_id, im_targets) in enumerate(scene_targets.items()):
            im_meta_data = [im_ind, (im_id, im_targets)]
            im_meta_datas.append(im_meta_data)

        # Calculate errors for each image in parallel.
        def calculate_errors_per_image(im_metaData):
            im_ind, (im_id, im_targets) = im_metaData
            im_errs = []
            per_image_ests_counter = 0
            if im_ind % 10 == 0:
                logger.info(
                    "Calculating error {} - method: {}, dataset: {}{}, scene: {}, "
                    "im: {}".format(
                        p["error_type"],
                        method,
                        dataset,
                        split_type_str,
                        scene_id,
                        im_ind,
                    )
                )

            # Try extracting either a K or a CameraModel from scene camera parameters 
            K = None
            cam = None
            if "cam_K" in scene_camera[im_id]:
                # Only Classic BOP19 format
                K = scene_camera[im_id]["cam_K"]
            elif htt_available:
                # Works with both Classic BOP19 and H3 BOP24 format
                cam = pose_error_htt.create_camera_model(scene_camera[im_id])
                if cam.distortion_model.__name__ == "PinholePlaneCameraModel":
                    K = cam.uv_to_window_matrix()
            elif ("cam_model" in scene_camera[im_id]
                  and scene_camera[im_id]["cam_model"] == "PinholePlaneCameraModel"):
                # Only H3 BOP24 format
                calib = scene_camera[im_id]["cam_model"]
                fx, fy, cx, cy = calib["projection_params"][:4]
                K = np.array([fx,0, cx,
                              0, fy,cy,
                              0, 0, 1]).reshape((3,3))

            if p["error_type"] in ['vsd','cus','proj']:
                assert K is not None, "Error type {} is not supported for non pinhole cameras".format(p["error_type"])
            if p["error_type"] in ["mspd"]:
                assert (K is not None) or (cam is not None), "Dataset {} requires Handa-Tracking-Toolkit as it is a non Pinhole camera (see bop_toolkit/README.md)".format(dataset)

            # Load the depth image if VSD is selected as the pose error function.
            depth_im = None
            if p["error_type"] == "vsd":
                depth_path = dp_split["depth_tpath"].format(
                    scene_id=scene_id, im_id=im_id
                )
                depth_im = inout.load_depth(depth_path)
                depth_im *= scene_camera[im_id]["depth_scale"]  # Convert to [mm].

            # Create im_targets directly from scene_gt and scene_gt_info.
            im_gt = scene_gt[im_id]
            im_gt_info = scene_gt_info[im_id]
            if p["eval_mode"] == "detection":
                # We need to re-define the target file for 6D detection tasks because:
                # 1. For BOP-Classic, the function of calculating object visibility has been changed, we cannot create the exact same number of `visib_count` as in target_filename _bop19.json from GT
                # so our unit tests with using prediction created from GT fails, and cannot get 100%. Re-loading the target objects from GT make sures the score will be 100%
                # 2. We want to consider all GT, not only GT>visib_gt_min since we want to ignore estimation matches with GT < visib_gt_min.  
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
                per_image_ests_counter += len(obj_ests_sorted)

                # Calculate error of each pose estimate w.r.t. all GT poses of the same
                # object class.
                for est_id, est in obj_ests_sorted:
                    # Estimated pose.
                    R_e = est["R"]
                    t_e = est["t"]

                    errs = {}  # Errors w.r.t. GT poses of the same object class.
                    gt_visib_fracts = {}
                    for gt_id, gt in enumerate(scene_gt[im_id]):
                        if gt["obj_id"] != obj_id:
                            continue

                        # Ground-truth pose.
                        R_g = gt["cam_R_m2c"]
                        t_g = gt["cam_t_m2c"]
                        gt_visib_fract = scene_gt_info[im_id][gt_id]["visib_fract"]

                        # Check if the projections of the bounding spheres of the object in
                        # the two poses overlap (to speed up calculation of some errors).
                        sphere_projections_overlap = None
                        if p["error_type"] in ["vsd", "cus"]:
                            radius = 0.5 * models_info[obj_id]["diameter"]
                            sphere_projections_overlap = (
                                misc.overlapping_sphere_projections(
                                    radius, t_e.squeeze(), t_g.squeeze()
                                )
                            )

                        # Check if the bounding spheres of the object in the two poses
                        # overlap (to speed up calculation of some errors).
                        spheres_overlap = None
                        if p["error_type"] in ["ad", "add", "adi", "mssd"]:
                            center_dist = np.linalg.norm(t_e - t_g)
                            spheres_overlap = (
                                center_dist < models_info[obj_id]["diameter"]
                            )

                        if p["error_type"] == "vsd":
                            if not sphere_projections_overlap:
                                e = [1.0] * len(p["vsd_taus"])
                            else:
                                if p["num_workers"] == 1:
                                    e = pose_error.vsd(
                                        R_e,
                                        t_e,
                                        R_g,
                                        t_g,
                                        depth_im,
                                        K,
                                        p["vsd_deltas"][dataset],
                                        p["vsd_taus"],
                                        p["vsd_normalized_by_diameter"],
                                        models_info[obj_id]["diameter"],
                                        ren,
                                        obj_id,
                                        "step",
                                    )
                                else:  # delayed the calculation of the error for renderer_batch
                                    e = pose_error.POSE_ERROR_VSD_ARGS(
                                        R_e=R_e,
                                        t_e=t_e,
                                        R_g=R_g,
                                        t_g=t_g,
                                        depth_im=depth_im,
                                        K=K,
                                        vsd_deltas=p["vsd_deltas"][dataset],
                                        vsd_taus=p["vsd_taus"],
                                        vsd_normalized_by_diameter=p[
                                            "vsd_normalized_by_diameter"
                                        ],
                                        diameter=models_info[obj_id]["diameter"],
                                        obj_id=obj_id,
                                        step="step",
                                    )

                        elif p["error_type"] == "mssd":
                            if not spheres_overlap:
                                e = [float("inf")]
                            else:
                                e = [
                                    pose_error.mssd(
                                        R_e,
                                        t_e,
                                        R_g,
                                        t_g,
                                        models[obj_id]["pts"],
                                        models_sym[obj_id],
                                    )
                                ]

                        elif p["error_type"] == "mspd":
                            if cam is not None:
                                e = [
                                    pose_error_htt.mspd(
                                        R_e,
                                        t_e,
                                        R_g,
                                        t_g,
                                        cam,
                                        models[obj_id]["pts"],
                                        models_sym[obj_id],
                                    )
                                ]
                            elif K is not None:
                                e = [
                                    pose_error.mspd(
                                        R_e,
                                        t_e,
                                        R_g,
                                        t_g,
                                        K,
                                        models[obj_id]["pts"],
                                        models_sym[obj_id],
                                    )
                                ]
                            else:
                                raise ValueError("Either 'K' or 'cam_model' should be defined at this point")

                        elif p["error_type"] in ["ad", "add", "adi"]:
                            if not spheres_overlap:
                                # Infinite error if the bounding spheres do not overlap. With
                                # typically used values of the correctness threshold for the AD
                                # error (e.g. k*diameter, where k = 0.1), such pose estimates
                                # would be considered incorrect anyway.
                                e = [float("inf")]
                            else:
                                if p["error_type"] == "ad":
                                    if obj_id in dp_model["symmetric_obj_ids"]:
                                        e = [
                                            pose_error.adi(
                                                R_e,
                                                t_e,
                                                R_g,
                                                t_g,
                                                models[obj_id]["pts"],
                                            )
                                        ]
                                    else:
                                        e = [
                                            pose_error.add(
                                                R_e,
                                                t_e,
                                                R_g,
                                                t_g,
                                                models[obj_id]["pts"],
                                            )
                                        ]

                                elif p["error_type"] == "add":
                                    e = [
                                        pose_error.add(
                                            R_e, t_e, R_g, t_g, models[obj_id]["pts"]
                                        )
                                    ]

                                else:  # 'adi'
                                    e = [
                                        pose_error.adi(
                                            R_e, t_e, R_g, t_g, models[obj_id]["pts"]
                                        )
                                    ]

                        elif p["error_type"] == "cus":
                            if sphere_projections_overlap:
                                e = [pose_error.cus(R_e, t_e, R_g, t_g, K, ren, obj_id)]
                            else:
                                e = [1.0]

                        elif p["error_type"] == "proj":
                            e = [
                                pose_error.proj(
                                    R_e, t_e, R_g, t_g, K, models[obj_id]["pts"]
                                )
                            ]

                        elif p["error_type"] == "rete":
                            e = [pose_error.re(R_e, R_g), pose_error.te(t_e, t_g)]

                        elif p["error_type"] == "re":
                            e = [pose_error.re(R_e, R_g)]

                        elif p["error_type"] == "te":
                            e = [pose_error.te(t_e, t_g)]

                        else:
                            raise ValueError("Unknown pose error function.")

                        errs[gt_id] = e
                        gt_visib_fracts[gt_id] = gt_visib_fract
                        
                    # Save the calculated errors.
                    im_errs.append(
                        {
                            "im_id": im_id,
                            "obj_id": obj_id,
                            "est_id": est_id,
                            "score": est["score"],
                            "errors": errs,
                            "scene_id": scene_id,
                            "gt_visib_fracts": gt_visib_fracts,
                        }
                    )
                assert (
                    len(im_errs) == per_image_ests_counter
                ), f"{len(im_errs)} != {per_image_ests_counter}"
            return im_errs

        scene_errs = []
        if p["num_workers"] == 1:
            for im_meta_data in im_meta_datas:
                im_errs = calculate_errors_per_image(im_meta_data)
                ests_counter += len(im_errs)
                scene_errs.extend(im_errs)
        else:
            pool = multiprocessing.Pool(p["num_workers"])
            all_im_errs = pool.map(calculate_errors_per_image, im_meta_datas)
            if p["error_type"] == "vsd":
                all_im_errs = ren.run_vsd(all_im_errs)
            for im_errs in all_im_errs:
                ests_counter += len(im_errs)
                scene_errs.extend(im_errs)

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
        if p["error_type"] == "vsd":
            # For VSD, save errors for each tau value to a different file.
            for vsd_tau_id, vsd_tau in enumerate(p["vsd_taus"]):
                error_sign = misc.get_error_signature(
                    p["error_type"],
                    p["n_top"],
                    vsd_delta=p["vsd_deltas"][dataset],
                    vsd_tau=vsd_tau,
                )

                # Keep only errors for the current tau.
                scene_errs_curr = copy.deepcopy(scene_errs)
                for err in scene_errs_curr:
                    for gt_id in err["errors"].keys():
                        err["errors"][gt_id] = [err["errors"][gt_id][vsd_tau_id]]

                save_errors(error_sign, scene_errs_curr)
        else:
            error_sign = misc.get_error_signature(p["error_type"], p["n_top"])
            save_errors(error_sign, scene_errs)

    time_total = time.time() - time_start
    logger.info(
        "Calculation of errors for {} estimates took {}s.".format(
            ests_counter, time_total
        )
    )

logger.info("Done.")
