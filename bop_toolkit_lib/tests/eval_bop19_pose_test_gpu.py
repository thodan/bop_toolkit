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
args = parser.parse_args()

# Define the input directory
INPUT_DIR = "./bop_toolkit_lib/tests/data/"

# Define the output directory
OUTPUT_DIR = "./bop_toolkit_lib/tests/logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the dataset dictionary
FILE_DICTIONARY = {
    "lmo_megaPose": "cnos-fastsammegapose_lmo-test_16ab01bd-f020-4194-9750-d42fc7f875d2.csv",
    "icbin_megaPose": "cnos-fastsammegapose_icbin-test_7c9f443f-b900-41bb-af01-09b8eddfc2c4.csv",
    "tudl_megaPose": "cnos-fastsammegapose_tudl-test_1328490c-bf88-46ce-a12c-a5e5a7712220.csv",
    "ycbv_megaPose": "cnos-fastsammegapose_ycbv-test_8fe0af14-16e3-431a-83e7-df00e93828a6.csv",
    "tless_megaPose": "cnos-fastsammegapose_tless-test_94e046a0-42af-495f-8a35-11ce8ee6f217.csv",
}


# read the file
for dataset_method_name, file_name in FILE_DICTIONARY.items():
    input_path = f"{INPUT_DIR}/{file_name}"
    output_filename = file_name
    ests = inout.load_bop_results(input_path, version="bop19")
    print(f"Using {dataset_method_name} with {len(ests)} instances")

# Loop through each entry in the dictionary and execute the command
for dataset_method_name, file_name in tqdm(
    FILE_DICTIONARY.items(), desc="Executing..."
):
    output_file_name = f"{OUTPUT_DIR}/eval_bop24_pose_test_{dataset_method_name}.txt"
    command = [
        "python",
        "scripts/eval_bop19_pose.py",
        "--renderer_type=vispy",
        "--results_path",
        INPUT_DIR,
        "--eval_path",
        INPUT_DIR,
        "--result_filenames",
        file_name,
        "--num_worker",
        "10",
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
