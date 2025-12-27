# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Shows BOP19 metrics and plots recall curves after running eval_bop19_pose.py"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from bop_toolkit_lib import config
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


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
    "visib_gt_min": -1,
    # Plot Recall curves
    "plot_recall_curves": True,
    # Names of files with results for which to calculate the errors (assumed to be
    # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
    # description of the format. Example results can be found at:
    # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip
    "result_filenames": [
        "/path/to/csv/with/results",
    ],
    # Folder for the calculated pose errors and performance scores.
    "eval_path": config.eval_path,
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--visib_gt_min", default=p["visib_gt_min"])
parser.add_argument(
    "--result_filenames",
    default=",".join(p["result_filenames"]),
    help="Comma-separated names of files with results.",
)
parser.add_argument(
    "--eval_path",
    default=p["eval_path"],
    help="Comma-separated names of files with results.",
)
args = parser.parse_args()

p["visib_gt_min"] = float(args.visib_gt_min)
p["result_filenames"] = args.result_filenames.split(",")


# Evaluation.
# ------------------------------------------------------------------------------
def plot_recall_curves(recall_dict, p):
    """Plots recall curves and displays BOP19 metrics

    :param recall_dict: dictionary containing bop19 recall results
    :param p: parameters from show_performance_bop19.py
    """

    for i, error in enumerate(p["errors"]):
        if error["type"] == "mspd":
            corr_thres = ["{}".format(e) for sl in error["correct_th"] for e in sl]
        else:
            corr_thres = ["{:.2f}".format(e) for sl in error["correct_th"] for e in sl]

        recalls = recall_dict[error["type"]]
        all_recalls = []
        plt.figure()

        for key in sorted(recalls):
            threshold = key.split("=")[-1]
            if "vsd" in key:
                plt.plot(recalls[key], label="tau: " + threshold)
            else:
                plt.plot(recalls[key])
            all_recalls += recalls[key]

        plt.legend()

        plt.xticks(np.arange(len(corr_thres)), corr_thres)
        plt.ylim([0, 1])
        plt.ylabel("recall")
        if error["type"] == "mspd":
            plt.xlabel("thres @ r px")
        else:
            plt.xlabel("thres @ object diameter")

        plt.title(
            error["type"]
            + " - "
            + "average recall: "
            + "{:.4f}".format(np.mean(all_recalls))
        )

    plt.show()

for result_filename in p["result_filenames"]:
    misc.log("===========")
    misc.log("SHOWING: {}".format(result_filename))
    misc.log("===========")

    time_start = time.time()
    aur = {}

    recall_dict = {e["type"]: {} for e in p["errors"]}

    for error in p["errors"]:
        # Name of the result and the dataset.
        result_name, _, dataset, _, _, _ = inout.parse_result_filename(result_filename)

        # Paths (rel. to config.eval_path) to folders with calculated pose errors.
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
                error_dir_paths[error_sign] = os.path.join(args.eval_path, result_name, error_sign)
        else:
            error_sign = misc.get_error_signature(error["type"], error["n_top"])
            error_dir_paths[error_sign] = os.path.join(args.eval_path, result_name, error_sign)

        # Recall scores for all settings of the threshold of correctness (and also
        # of the misalignment tolerance tau in the case of VSD).
        recalls = []

        # Calculate performance scores.
        for error_sign, error_dir_path in error_dir_paths.items():
            recall_dict[error["type"]][error_sign] = []
            for correct_th in error["correct_th"]:
                # Path to file with calculated scores.
                score_sign = misc.get_score_signature(correct_th, p["visib_gt_min"])

                scores_filename = "scores_{}.json".format(score_sign)
                scores_path = os.path.join(
                    args.eval_path, result_name, error_sign, scores_filename
                )

                # Load the scores.
                misc.log("Loading calculated scores from: {}".format(scores_path))
                scores = inout.load_json(scores_path)
                recalls.append(scores["recall"])
                recall_dict[error["type"]][error_sign].append(scores["recall"])

        # Area under the recall surface/curve.
        aur[error["type"]] = np.mean(recalls)

    time_total = time.time() - time_start

    # output final scores and plot recall curves
    err_types = [e["type"] for e in p["errors"]]
    for err_type in err_types:
        misc.log("Average Recall {}: {}".format(err_type, aur[err_type]))

    if set(["vsd", "mssd", "mspd"]).issubset(err_types):
        test_set = os.path.basename(result_filename)
        mean_error = np.mean([aur[err_type] for err_type in err_types])
        misc.log("Average BOP score on {}: {}".format(test_set, mean_error))

    if p["plot_recall_curves"]:
        plot_recall_curves(recall_dict, p)

misc.log("Done.")
