#!/usr/bin/env python
import os
import re
import time
import shutil
import subprocess
import numpy as np
from tqdm import tqdm
import argparse

from bop_toolkit_lib import inout
from bop_toolkit_lib import config
from bop_toolkit_lib import misc


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
    # Use generate results from gt files instead of submissions 
    "gt_from_datasets": [],  # e.g. ['ycbv', 'lmo']
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    "use_gpu": config.use_gpu,  # Use torch for the calculation of errors.
    "num_workers": config.num_workers,  # Number of parallel workers for the calculation of errors.
    "tolerance": 1e-3,  # tolerance between expected scores and evaluated ones.
}

parser = argparse.ArgumentParser()
parser.add_argument("--gt_from_datasets", default="", help='Comma separated list of dataset names, e.g. "ycbv,tless,lmo"', type=str)
parser.add_argument("--renderer_type", default=p["renderer_type"])
parser.add_argument("--use_gpu", action="store_true", default=p["use_gpu"])
parser.add_argument("--num_workers", default=p["num_workers"])
parser.add_argument("--tolerance", default=p["tolerance"], type=float)
parser.add_argument("--num_false_positives", default=0, type=int)
args = parser.parse_args()

p["renderer_type"] = str(args.renderer_type)
p["gt_from_datasets"] = args.gt_from_datasets.split(',') if len(args.gt_from_datasets) > 0 else [] 
p["num_workers"] = int(args.num_workers)
p["use_gpu"] = bool(args.use_gpu)
p["tolerance"] = float(args.tolerance)


RESULT_PATH = "./bop_toolkit_lib/tests/data/results_sub"
EVAL_PATH = "./bop_toolkit_lib/tests/data/eval"
LOGS_PATH = "./bop_toolkit_lib/tests/data/logs"
os.makedirs(EVAL_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)


# Define the dataset dictionary
FILE_DICTIONARY = {
    "lmo_megaPose": ("cnos-fastsammegapose_lmo-test_16ab01bd-f020-4194-9750-d42fc7f875d2.csv", "test_targets_bop19.json"),
    "lmo_gt": ("gt-pbrreal-rgb-mmodel_lmo-test_lmo.csv", "test_targets_bop19.json"),
    "tless_megaPose": ("cnos-fastsammegapose_tless-test_94e046a0-42af-495f-8a35-11ce8ee6f217.csv", "test_targets_bop19.json"),
    "tless_gt": ("gt-pbrreal-rgb-mmodel_tless-test_tless.csv", "test_targets_bop19.json"),
}

# Define the expected scores (from bop24 bop_toolkit commit e7ba9f2)
EXPECTED_OUTPUT = {
    "lmo_megaPose": {
        "bop24_mAP_mssd": 0.44000810549674874,
        "bop24_mAP_mspd": 0.5685263874253823,
        "bop24_mAP": 0.5042672464610656,
    },
    "lmo_gt": {
        "bop24_mAP_mssd": 0.9678217821782178,
        "bop24_mAP_mspd": 0.9678217821782178,
        "bop24_mAP": 0.9678217821782178,
    },
    "tless_megaPose": {
        "bop24_mAP_mssd": 0.4387896135435956,
        "bop24_mAP_mspd": 0.5085668907013268,
        "bop24_mAP": 0.4736782521224612,
    },
    "tless_gt": {
        "bop24_mAP_mssd": 0.9765676567656766,
        "bop24_mAP_mspd": 0.9765676567656766,
        "bop24_mAP": 0.9765676567656766,
    },
}

# If using ground truth datasets, redefine result files and expected results
if len(p["gt_from_datasets"]) > 0:
    RESULT_PATH = "./bop_toolkit_lib/tests/data/results_gt"
    # assuming all concerned datasets are bop24
    FILE_DICTIONARY = {
        f"{ds}_gt": (f"gt-results_{ds}-test_pose.csv", "test_targets_bop24.json")
        for ds in p["gt_from_datasets"]
    }
    EXPECTED_OUTPUT = {
        f"{ds}_gt": {
            "bop24_mAP_mssd": 1.0,
            "bop24_mAP_mspd": 1.0,
            "bop24_mAP": 1.0,
        }
        for ds in p["gt_from_datasets"]
    }



# read the file
for dataset_method_name, (file_name, test_targets_name) in FILE_DICTIONARY.items():
    result_file_path = f"{RESULT_PATH}/{file_name}"
    output_filename = file_name.replace(
        ".csv", f"_{args.num_false_positives}_false_positives.csv"
    )
    ests = inout.load_bop_results(result_file_path, version="bop19")
    if args.num_false_positives > 0:
        # create dummy estimates
        dummy_ests = []
        for i in range(args.num_false_positives):
            est = ests[i % len(ests)].copy()
            est["R"] = np.eye(3)
            est["t"] = np.ones(3)
            dummy_ests.append(est)
        ests.extend(dummy_ests)
        inout.save_bop_results(f"{RESULT_PATH}/{output_filename}", ests, version="bop19")
        FILE_DICTIONARY[dataset_method_name] = output_filename
        misc.log(
            f"Added {args.num_false_positives} false positives to {dataset_method_name} (total: {len(ests)} instances)"
        )
    else:
        misc.log(f"Using {dataset_method_name} with {len(ests)} instances")

# Loop through each entry in the dictionary and execute the command
for dataset_method_name, (file_name, test_targets_name) in tqdm(
    FILE_DICTIONARY.items(), desc="Executing..."
):
    log_file_path = f"{LOGS_PATH}/eval_bop24_pose_test_{dataset_method_name}.txt"
    # Remove eval sub path to start clean
    eval_path_dir = os.path.join(EVAL_PATH, file_name.split('.')[0])
    if os.path.exists(eval_path_dir):
        shutil.rmtree(eval_path_dir)
    command = [
        "python",
        "scripts/eval_bop24_pose.py",
        "--renderer_type",
        p["renderer_type"],
        "--results_path",
        RESULT_PATH,
        "--eval_path",
        EVAL_PATH,
        "--result_filenames",
        file_name,
        "--num_worker",
        str(p["num_workers"]),
        "--targets_filename",
        test_targets_name
    ]
    if p["use_gpu"]:
        command.append("--use_gpu")
    command_str = " ".join(command)
    misc.log(f"Executing: {command_str}")
    start_time = time.perf_counter()
    with open(log_file_path, "a") as output_file:
        returncode = subprocess.run(command, stdout=output_file, stderr=subprocess.STDOUT).returncode
        if returncode != 0:
            misc.log('FAILED: '+command_str)    
    end_time = time.perf_counter()
    misc.log(f"Evaluation time for {dataset_method_name}: {end_time - start_time} seconds")

misc.log("Script executed successfully.")


# Check scores for each dataset
if args.num_false_positives == 0:
    for dataset_name, _ in tqdm(FILE_DICTIONARY.items(), desc="Verifying..."):
        if dataset_name in EXPECTED_OUTPUT:
            log_file_path = f"{LOGS_PATH}/eval_bop24_pose_test_{dataset_name}.txt"

            # Read the content of the log file
            with open(log_file_path, "r") as log_file:
                last_lines = log_file.readlines()[-7:]

            # Combine the last lines into a single string
            log_content = "".join(last_lines)

            # Extract scores using regular expressions
            scores = {
                key: float(value)
                for key, value in re.findall(r"- (\S+): (\S+)", log_content)
            }

            # Compare the extracted scores with the expected scores
            for key, expected_value in EXPECTED_OUTPUT[dataset_name].items():
                actual_value = scores.get(key)
                if actual_value is not None:
                    if abs(actual_value - expected_value) < p["tolerance"]:
                        misc.log(f"{dataset_name}: {key} - PASSED")
                    else:
                        misc.log(
                            f"{dataset_name}: {key} - FAILED. Expected: {expected_value}, Actual: {actual_value}"
                        )
                else:
                    misc.log(f"{dataset_name}: {key} - NOT FOUND")
                    misc.log(f"Please check the log file {log_file_path} for more details.")
    misc.log("Verification completed.")
