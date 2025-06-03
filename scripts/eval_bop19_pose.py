# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Evaluation script for the BOP Challenge 2019/2020."""

import os
import time
import argparse
import multiprocessing
import subprocess
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

# Get the base name of the file without the .py extension
file_name = os.path.splitext(os.path.basename(__file__))[0]
logger = misc.get_logger(file_name)

# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
    # Errors to calculate.
    "errors": [
        {
            "n_top": -1,
            "type": "vsd",
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
            "correct_th": [[th] for th in np.arange(0.05, 0.51, 0.05)],
        },
        {
            "n_top": -1,
            "type": "mssd",
            "correct_th": [[th] for th in np.arange(0.05, 0.51, 0.05)],
        },
        {
            "n_top": -1,
            "type": "mspd",
            "correct_th": [[th] for th in np.arange(5, 51, 5)],
        },
    ],
    # Minimum visible surface fraction of a valid GT pose.
    # by default, we consider only objects that are at least 10% visible
    "visib_gt_min": -1,
    # See misc.get_symmetry_transformations().
    "max_sym_disc_step": 0.01,
    # Type of the renderer (used for the VSD pose error function).
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    # Names of files with results for which to calculate the errors (assumed to be
    # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
    # description of the format. Example results can be found at:
    # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip
    "result_filenames": [
        "/relative/path/to/csv/with/results",
    ],
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder for the calculated pose errors and performance scores.
    "eval_path": config.eval_path,
    # File with a list of estimation targets to consider. The file is assumed to
    # be stored in the dataset folder.
    "targets_filename": "test_targets_bop19.json",
    "num_workers": config.num_workers,  # Number of parallel workers for the calculation of errors.
    "use_gpu": config.use_gpu,  # Use torch for the calculation of errors.
    "device": "cuda:0",  # if use_gpu is true, use "device" for torch computations.
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--renderer_type", type=str, default=p["renderer_type"])
parser.add_argument(
    "--result_filenames",
    type=str,
    default=",".join(p["result_filenames"]),
    help="Comma-separated names of files with results.",
)
parser.add_argument("--results_path", type=str, default=p["results_path"])
parser.add_argument("--eval_path", type=str, default=p["eval_path"])
parser.add_argument("--targets_filename", type=str, default=p["targets_filename"])
parser.add_argument("--num_workers", type=int, default=p["num_workers"])
parser.add_argument("--use_gpu", action="store_true", default=p["use_gpu"])
parser.add_argument("--device", type=str, default=p["device"])
args = parser.parse_args()

result_filenames = args.result_filenames.split(",")

eval_time_start = time.time()
# Evaluation.
# ------------------------------------------------------------------------------
for result_filename in result_filenames:
    logger.info("===========")
    logger.info("EVALUATING: {}".format(result_filename))
    logger.info("===========")

    time_start = time.time()

    # Volume under recall surface (VSD) / area under recall curve (MSSD, MSPD).
    average_recalls = {}

    # Name of the result and the dataset.
    result_name, _, dataset, _, _, _ = inout.parse_result_filename(result_filename)

    # Load and check results, calculate the average estimation time per image.
    result_path = os.path.join(args.results_path, result_filename)
    ests = inout.load_bop_results(result_path, version="bop19")
    check_passed, check_msg, times, times_available = inout.check_consistent_timings(ests, "im_id")
    if not check_passed:
        raise ValueError(check_msg)
    average_time_per_image = np.mean(list(times.values())) if times_available else -1.0

    # Evaluate the pose estimates.
    for error in p["errors"]:
        # Calculate error of the pose estimates.
        calc_error_script, is_gpu_script_used = misc.get_eval_calc_errors_script_name(args.use_gpu, error["type"], dataset)
        calc_errors_cmd = [
            "python",
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                calc_error_script,
            ),
            "--n_top={}".format(error["n_top"]),
            "--error_type={}".format(error["type"]),
            "--result_filenames={}".format(result_filename),
            "--renderer_type={}".format(args.renderer_type),
            "--results_path={}".format(args.results_path),
            "--eval_path={}".format(args.eval_path),
            "--targets_filename={}".format(args.targets_filename),
            "--max_sym_disc_step={}".format(p["max_sym_disc_step"]),
            "--skip_missing=1",
            "--num_workers={}".format(args.num_workers),
        ]
        if is_gpu_script_used:
            calc_errors_cmd.append(f"--device={args.device}")
        if error["type"] == "vsd":
            vsd_deltas_str = ",".join(
                ["{}:{}".format(k, v) for k, v in error["vsd_deltas"].items()]
            )
            calc_errors_cmd += [
                "--vsd_deltas={}".format(vsd_deltas_str),
                "--vsd_taus={}".format(",".join(map(str, error["vsd_taus"]))),
                "--vsd_normalized_by_diameter={}".format(
                    error["vsd_normalized_by_diameter"]
                ),
            ]

        logger.info("Running: " + " ".join(calc_errors_cmd))
        if subprocess.call(calc_errors_cmd) != 0:
            raise RuntimeError("Calculation of pose errors failed.")

        # Paths (rel. to p['eval_path']) to folders with calculated pose errors.
        # For VSD, there is one path for each setting of tau. For the other pose
        # error functions, there is only one path.
        error_dir_paths = {}
        if error["type"] == "vsd":
            for vsd_tau in error["vsd_taus"]:
                error_sign = misc.get_error_signature(
                    error["type"],
                    error["n_top"],
                    vsd_delta=error["vsd_deltas"][dataset],
                    vsd_tau=vsd_tau,
                )
                error_dir_paths[error_sign] = os.path.join(result_name, error_sign)
        else:
            error_sign = misc.get_error_signature(error["type"], error["n_top"])
            error_dir_paths[error_sign] = os.path.join(result_name, error_sign)

        # Recall scores for all settings of the threshold of correctness (and also
        # of the misalignment tolerance tau in the case of VSD).

        calc_scores_cmds = []
        # Calculate performance scores.
        for error_sign, error_dir_path in error_dir_paths.items():
            for correct_th in error["correct_th"]:
                calc_scores_cmd = [
                    "python",
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "eval_calc_scores.py",
                    ),
                    "--error_dir_paths={}".format(error_dir_path),
                    "--eval_path={}".format(args.eval_path),
                    "--targets_filename={}".format(args.targets_filename),
                    "--visib_gt_min={}".format(p["visib_gt_min"]),
                ]

                calc_scores_cmd += [
                    "--correct_th_{}={}".format(
                        error["type"], ",".join(map(str, correct_th))
                    )
                ]
                calc_scores_cmds.append(calc_scores_cmd)

        if args.num_workers == 1:
            for calc_scores_cmd in calc_scores_cmds:
                logger.info("Running: " + " ".join(calc_scores_cmd))
                if subprocess.call(calc_scores_cmd) != 0:
                    raise RuntimeError("Calculation of performance scores failed.")
        else:
            with multiprocessing.Pool(args.num_workers) as pool:
                pool.map_async(misc.run_command, calc_scores_cmds)
                pool.close()
                pool.join()

        recalls = []
        for error_sign, error_dir_path in error_dir_paths.items():
            for correct_th in error["correct_th"]:
                # Path to file with calculated scores.
                score_sign = misc.get_score_signature(correct_th, p["visib_gt_min"])

                scores_filename = "scores_{}.json".format(score_sign)
                scores_path = os.path.join(
                    args.eval_path, result_name, error_sign, scores_filename
                )

                # Load the scores.
                logger.info("Loading calculated scores from: {}".format(scores_path))
                scores = inout.load_json(scores_path)
                recalls.append(scores["recall"])

        average_recalls[error["type"]] = np.mean(recalls)

        logger.info("Recall scores: {}".format(" ".join(map(str, recalls))))
        logger.info("Average recall: {}".format(average_recalls[error["type"]]))

    time_total = time.time() - time_start
    logger.info("Evaluation of {} took {}s.".format(result_filename, time_total))

    # Calculate the final scores.
    final_scores = {}
    for error in p["errors"]:
        final_scores["bop19_average_recall_{}".format(error["type"])] = average_recalls[
            error["type"]
        ]

    # Final score for the given dataset.
    final_scores["bop19_average_recall"] = np.mean(
        [average_recalls["vsd"], average_recalls["mssd"], average_recalls["mspd"]]
    )

    # Average estimation time per image.
    final_scores["bop19_average_time_per_image"] = average_time_per_image

    # Save the final scores.
    final_scores_path = os.path.join(args.eval_path, result_name, "scores_bop19.json")
    inout.save_json(final_scores_path, final_scores)

    # Print the final scores.
    logger.info("FINAL SCORES:")
    for score_name, score_value in final_scores.items():
        logger.info("- {}: {}".format(score_name, score_value))

total_eval_time = time.time() - eval_time_start
logger.info("Evaluation took {}s.".format(total_eval_time))
logger.info("Done.")
