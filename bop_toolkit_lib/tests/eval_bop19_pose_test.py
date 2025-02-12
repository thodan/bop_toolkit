#!/usr/bin/env python
import os
import re
import time
import shutil
import subprocess
import argparse
from tqdm import tqdm

from bop_toolkit_lib import config
from bop_toolkit_lib import misc


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
    # Use generate results from gt files instead of submissions 
    "gt_from_datasets": [],  # e.g. ['ycbv', 'lmo']
    "targets_filename": "test_targets_bop19.json",
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    "use_gpu": config.use_gpu,  # Use torch for the calculation of errors.
    "num_workers": config.num_workers,  # Number of parallel workers for the calculation of errors.
    "tolerance": 1e-3,  # tolerance between expected scores and evaluated ones.
}

parser = argparse.ArgumentParser()
parser.add_argument("--gt_from_datasets", default="", help='Comma separated list of dataset names, e.g. "ycbv,tless,lmo"', type=str)
parser.add_argument("--targets_filename", default=p["targets_filename"])
parser.add_argument("--renderer_type", default=p["renderer_type"])
parser.add_argument("--use_gpu", action="store_true", default=p["use_gpu"])
parser.add_argument("--num_workers", default=p["num_workers"])
parser.add_argument("--tolerance", default=p["tolerance"], type=float)
args = parser.parse_args()

p["renderer_type"] = str(args.renderer_type)
p["gt_from_datasets"] = args.gt_from_datasets.split(',') if len(args.gt_from_datasets) > 0 else [] 
p["targets_filename"] = str(args.targets_filename)
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
    "lmo_megaPose": "cnos-fastsammegapose_lmo-test_16ab01bd-f020-4194-9750-d42fc7f875d2.csv",
    "icbin_megaPose": "cnos-fastsammegapose_icbin-test_7c9f443f-b900-41bb-af01-09b8eddfc2c4.csv",
    "tudl_megaPose": "cnos-fastsammegapose_tudl-test_1328490c-bf88-46ce-a12c-a5e5a7712220.csv",
    "ycbv_megaPose": "cnos-fastsammegapose_ycbv-test_8fe0af14-16e3-431a-83e7-df00e93828a6.csv",
    "tless_megaPose": "cnos-fastsammegapose_tless-test_94e046a0-42af-495f-8a35-11ce8ee6f217.csv",
}

# Define the expected scores
# megapose-CNOS_fast (https://bop.felk.cvut.cz/sub_info/4299/)
EXPECTED_OUTPUT = {
    "lmo_megaPose": {
        "bop19_average_recall_vsd": 0.3976885813148789,
        "bop19_average_recall_mssd": 0.49411764705882344,
        "bop19_average_recall_mspd": 0.6067128027681661,
        "bop19_average_recall": 0.49950634371395614,
    },
    "icbin_megaPose": {
        "bop19_average_recall_vsd": 0.33743001,
        "bop19_average_recall_mssd": 0.36108623,
        "bop19_average_recall_mspd": 0.40184770,
        "bop19_average_recall": 0.36678798,
    },
    "tudl_megaPose": {
        "bop19_average_recall_vsd": 0.52866667,
        "bop19_average_recall_mssd": 0.62200000,
        "bop19_average_recall_mspd": 0.80766667,
        "bop19_average_recall": 0.65277778,
    },
    "ycbv_megaPose": {
        "bop19_average_recall_vsd": 0.52073733,
        "bop19_average_recall_mssd": 0.57756488,
        "bop19_average_recall_mspd": 0.70538443,
        "bop19_average_recall": 0.60122888,
    },
    "tless_megaPose": {
        "bop19_average_recall_vsd": 0.4574871555347968,
        "bop19_average_recall_mssd": 0.45304374902693445,
        "bop19_average_recall_mspd": 0.5210493538844776,
        "bop19_average_recall": 0.47719341948206956,
    },
}

# If using ground truth datasets, redefine result files and expected results
if len(p["gt_from_datasets"]) > 0:
    RESULT_PATH = "./bop_toolkit_lib/tests/data/results_gt"
    FILE_DICTIONARY = {
        f"{ds}_gt": f"gt-results_{ds}-test_pose.csv"
        for ds in p["gt_from_datasets"]
    }
    EXPECTED_OUTPUT = {
        f"{ds}_gt": {
            "bop19_average_recall_vsd": 1.0,
            "bop19_average_recall_mssd": 1.0,
            "bop19_average_recall_mspd": 1.0,
            "bop19_average_recall": 1.0,
        }
        for ds in p["gt_from_datasets"]
    }


assert FILE_DICTIONARY.keys() == EXPECTED_OUTPUT.keys()

# Loop through each entry in the dictionary and execute the command
for dataset_method_name, file_name in tqdm(
    FILE_DICTIONARY.items(), desc="Executing..."
):
    log_file_path = f"{LOGS_PATH}/eval_bop19_pose_test_{dataset_method_name}.txt"
    # Remove eval sub path to start clean
    eval_path_dir = os.path.join(EVAL_PATH, file_name.split('.')[0])
    if os.path.exists(eval_path_dir):
        shutil.rmtree(eval_path_dir)
    command = [
        "python",
        "scripts/eval_bop19_pose.py",
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
        p["targets_filename"],
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


# Check scores for each dataset
for dataset_method_name, _ in tqdm(FILE_DICTIONARY.items(), desc="Verifying..."):
    log_file_path = f"{LOGS_PATH}/eval_bop19_pose_test_{dataset_method_name}.txt"

    # Read the content of the log file
    with open(log_file_path, "r") as log_file:
        last_lines = log_file.readlines()[-7:]

    # Combine the last lines into a single string
    log_content = "".join(last_lines)

    # Extract scores using regular expressions
    scores = {
        key: float(value) for key, value in re.findall(r"- (\S+): (\S+)", log_content)
    }

    # Compare the extracted scores with the expected scores
    for key, expected_value in EXPECTED_OUTPUT[dataset_method_name].items():
        actual_value = scores.get(key)
        if actual_value is not None:
            if abs(actual_value - expected_value) < p["tolerance"]:
                misc.log(f"{dataset_method_name}: {key}: {actual_value} - PASSED")
            else:
                misc.log(
                    f"{dataset_method_name}: {key} - FAILED. Expected: {expected_value}, Actual: {actual_value}"
                )
        else:
            misc.log(f"{dataset_method_name}: {key} - NOT FOUND")

misc.log("Verification completed.")
