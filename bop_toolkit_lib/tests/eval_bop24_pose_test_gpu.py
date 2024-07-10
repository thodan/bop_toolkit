#!/usr/bin/env python
import subprocess
import os
import re
import time
import numpy as np
from tqdm import tqdm
import argparse
from bop_toolkit_lib import inout

parser = argparse.ArgumentParser()
parser.add_argument("--num_false_positives", default=0, type=int)
args = parser.parse_args()

# Define the input directory
INPUT_DIR = "./bop_toolkit_lib/tests/data/"

# Define the output directory
OUTPUT_DIR = "./bop_toolkit_lib/tests/logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the dataset dictionary
FILE_DICTIONARY = {
    "lmo_megaPose": "cnos-fastsammegapose_lmo-test_16ab01bd-f020-4194-9750-d42fc7f875d2.csv",
    "lmo_gt": "gt-pbrreal-rgb-mmodel_lmo-test_lmo.csv",
    "tless_megaPose": "cnos-fastsammegapose_tless-test_94e046a0-42af-495f-8a35-11ce8ee6f217.csv",
    "tless_gt": "gt-pbrreal-rgb-mmodel_tless-test_tless.csv",
}


# read the file
for dataset_method_name, file_name in FILE_DICTIONARY.items():
    input_path = f"{INPUT_DIR}/{file_name}"
    output_filename = file_name.replace(
        ".csv", f"_{args.num_false_positives}_false_positives_gpu.csv"
    )
    ests = inout.load_bop_results(input_path, version="bop19")
    if args.num_false_positives > 0:
        # create dummy estimates
        dummy_ests = []
        for i in range(args.num_false_positives):
            est = ests[i % len(ests)].copy()
            est["R"] = np.eye(3)
            est["t"] = np.ones(3)
            dummy_ests.append(est)
        ests.extend(dummy_ests)
        inout.save_bop_results(f"{INPUT_DIR}/{output_filename}", ests, version="bop19")
        FILE_DICTIONARY[dataset_method_name] = output_filename
        print(
            f"Added {args.num_false_positives} false positives to {dataset_method_name} (total: {len(ests)} instances)"
        )
    else:
        print(f"Using {dataset_method_name} with {len(ests)} instances")

EXPECTED_OUTPUT = {
    "lmo_megaPose": {
        "bop24_mAP_mssd": 0.4248351626966518,
        "bop24_mAP_mspd": 0.45256928969533605,
        "bop24_mAP": 0.4387022261959939,
    },
    "lmo_gt": {
        "bop24_mAP_mssd": 1.0,
        "bop24_mAP_mspd": 1.0,
        "bop24_mAP": 1.0,
    },
    "tless_megaPose": {
        "bop24_mAP_mssd": 0.3145410625507144,
        "bop24_mAP_mspd": 0.33137928347498696,
        "bop24_mAP": 0.3229601730128507,
    },
    "tless_gt": {
        "bop24_mAP_mssd": 1.0,  # 0.9791824358328837,
        "bop24_mAP_mspd": 1.0,  # 0.9791824358328837,
        "bop24_mAP": 1.0,  # 0.9791824358328837,
    },
}

# Loop through each entry in the dictionary and execute the command
for dataset_method_name, file_name in tqdm(
    FILE_DICTIONARY.items(), desc="Executing..."
):
    output_file_name = (
        f"{OUTPUT_DIR}/eval_bop24_pose_test_{dataset_method_name}_gpu.txt"
    )
    command = [
        "python",
        "scripts/eval_bop24_pose.py",
        "--renderer_type=vispy",
        "--results_path",
        INPUT_DIR,
        "--eval_path",
        INPUT_DIR,
        "--result_filenames",
        file_name,
        "--num_worker",
        "1",
    ]
    command.append("--use_gpu")
    command_ = " ".join(command)
    print(f"Executing: {command_}")
    start_time = time.time()
    with open(output_file_name, "a") as output_file:
        subprocess.run(command, stdout=output_file, stderr=subprocess.STDOUT)
    end_time = time.time()
    print(f"Execution time for {dataset_method_name}: {end_time - start_time} seconds")

print("Script executed successfully.")


# Check scores for each dataset
if args.num_false_positives == 0:
    for dataset_name, _ in tqdm(FILE_DICTIONARY.items(), desc="Verifying..."):
        if dataset_name in EXPECTED_OUTPUT:
            log_file_path = f"{OUTPUT_DIR}/eval_bop24_pose_test_{dataset_name}.txt"

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
                    if actual_value == expected_value:
                        print(f"{dataset_name}: {key} - PASSED")
                    else:
                        print(
                            f"{dataset_name}: {key} - FAILED. Expected: {expected_value}, Actual: {actual_value}"
                        )
                else:
                    print(f"{dataset_name}: {key} - NOT FOUND")
    print("Verification completed.")
