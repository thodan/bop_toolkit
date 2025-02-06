#!/usr/bin/env python
import os
import time
import shutil
import subprocess
from tqdm import tqdm
import argparse

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
    # Use generate results from gt files instead of submissions 
    "gt_from_datasets": [],  # e.g. ['ycbv', 'lmo']
    "tolerance": 1e-3,  # tolerance between expected scores and evaluated ones.
}

parser = argparse.ArgumentParser()
parser.add_argument("--gt_from_datasets", default="", help='Comma separated list of dataset names, e.g. "ycbv,tless,lmo"', type=str)
parser.add_argument("--tolerance", default=p["tolerance"], type=float)
args = parser.parse_args()

p["gt_from_datasets"] = args.gt_from_datasets.split(',') if len(args.gt_from_datasets) > 0 else [] 

# Define path to directories
RESULT_PATH = "./bop_toolkit_lib/tests/data/results_sub"
EVAL_PATH = "./bop_toolkit_lib/tests/data/eval"
LOGS_PATH = "./bop_toolkit_lib/tests/data/logs"
os.makedirs(EVAL_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# Define the dataset dictionary
# tuples: (submission name, annotation type, compressed)
FILE_DICTIONARY = {
    "ycbv_zebra_segm": ("zebraposesat-effnetb4_ycbv-test_5ed0eecc-96f8-498b-9438-d586d4d92528", "segm", False),  # https://bop.felk.cvut.cz/sub_info/3131/
    "ycbv_gdrnppdet_bbox": ("gdrnppdet-pbrreal_ycbv-test_abe6c5f1-cb26-4bbd-addc-bb76dd722a96", "bbox", True),  # https://bop.felk.cvut.cz/sub_info/2743/
}

# From BOP website
EXPECTED_OUTPUT = {
    "ycbv_zebra_segm": {
        "AP":	0.740,
        "AP50":	0.983,
        "AP75":	0.866,
        "AP_large":	0.767,
        "AP_medium":	0.697,
        "AP_small":	0.000,
        "AR1":	0.767,
        "AR10":	0.790,
        "AR100":	0.790,
        "AR_large":	0.828,
        "AR_medium":	0.697,
        "AR_small":	0.000,
    },
    "ycbv_gdrnppdet_bbox": {
        "AP":	0.852,
        "AP50":	0.992,
        "AP75":	0.981,
        "AP_large":	0.875,
        "AP_medium":	0.791,
        "AP_small":	0.050,
        "AR1":	0.880,
        "AR10":	0.880,
        "AR100":	0.880,
        "AR_large":	0.907,
        "AR_medium":	0.804,
        "AR_small":	0.300,
    }
}

# If using ground truth datasets, redefine result files and expected results
if len(p["gt_from_datasets"]) > 0:
    RESULT_PATH = "./bop_toolkit_lib/tests/data/results_gt"
    FILE_DICTIONARY = {
        f"{ds}_gt": (f"gt-results_{ds}-test_coco", "bbox", False)
        for ds in p["gt_from_datasets"]
    }
    EXPECTED_OUTPUT = {
        f"{ds}_gt": {
            "AP":	1.0,
            "AP50":	1.0,
            "AP75":	1.0,
            "AP_large":	1.0,
            "AP_medium":	1.0,
            "AP_small":	1.0,
            "AR1":	1.0,
            "AR10":	1.0,
            "AR100":	1.0,
            "AR_large":	1.0,
            "AR_medium":	1.0,
            "AR_small":	1.0,
        }
        for ds in p["gt_from_datasets"]
    }


# Loop through each entry in the dictionary and execute the command
for dataset_method_name, (sub_name, ann_type, compressed) in tqdm(
    FILE_DICTIONARY.items(), desc="Executing..."
):
    ext = ".json.gz" if compressed else ".json"
    result_filename = sub_name + ext
    log_file_path = f"{LOGS_PATH}/eval_bop22_coco_test_{dataset_method_name}.txt"
    # Remove eval sub path to start clean
    eval_path_dir = os.path.join(EVAL_PATH, sub_name)
    if os.path.exists(eval_path_dir):
        shutil.rmtree(eval_path_dir)
    command = [
        "python",
        "scripts/eval_bop22_coco.py",
        "--results_path", RESULT_PATH,
        "--eval_path", EVAL_PATH,
        "--result_filenames", result_filename,
        "--bbox_type", "amodal",
        "--ann_type", ann_type
    ]
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
for sub_short_name, (sub_name, ann_type, compressed) in tqdm(FILE_DICTIONARY.items(), desc="Verifying..."):
    if sub_short_name in EXPECTED_OUTPUT:
        eval_filename = f"scores_bop22_coco_{ann_type}.json"
        eval_file_path = os.path.join(EVAL_PATH, sub_name, eval_filename)
        eval_scores = inout.load_json(eval_file_path)
        for key, expected_score in EXPECTED_OUTPUT[sub_short_name].items():
            eval_score = eval_scores.get(key)
            if eval_score is not None:
                if abs(eval_score - expected_score) < args.tolerance:
                    misc.log(f"{sub_short_name}: {key} - PASSED")
                else:
                    misc.log(
                        f"{sub_short_name}: {key} - FAILED. Expected: {expected_score}, Actual: {eval_score}"
                    )
            else:
                misc.log(f"{sub_short_name}: {key} - NOT FOUND")
                misc.log(f"Please check the log file {log_file_path} and {eval_file_path} for more details.")

misc.log("Verification completed.")
