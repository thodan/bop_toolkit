#!/usr/bin/env python
import subprocess
import os
import re
from tqdm import tqdm

# Define the input directory
INPUT_DIR = "./bop_toolkit_lib/tests/data/"

# Define the output directory
OUTPUT_DIR = "./bop_toolkit_lib/tests/logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the dataset dictionary
FILE_DICTIONARY = {
    "icbin": "cnos-fastsammegapose_icbin-test_7c9f443f-b900-41bb-af01-09b8eddfc2c4.csv",
    "tudl": "cnos-fastsammegapose_tudl-test_1328490c-bf88-46ce-a12c-a5e5a7712220.csv",
    "lmo": "cnos-fastsammegapose_lmo-test_16ab01bd-f020-4194-9750-d42fc7f875d2.csv",
    # "ycbv": "cnos-fastsammegapose_ycbv-test_8fe0af14-16e3-431a-83e7-df00e93828a6.csv",
    # "tless": "cnos-fastsammegapose_tless-test_94e046a0-42af-495f-8a35-11ce8ee6f217.csv",
    # Add more entries as needed
}

EXPECTED_OUTPUT = {
    "icbin": {
        "bop19_average_recall_vsd": 0.337889137737962,
        "bop19_average_recall_mssd": 0.36108622620380737,
        "bop19_average_recall_mspd": 0.4018477043673013,
        "bop19_average_recall": 0.36694102276969015,
    },
    "tudl": {
        "bop19_average_recall_vsd": 0.5286500000000001,
        "bop19_average_recall_mssd": 0.6220000000000001,
        "bop19_average_recall_mspd": 0.8076666666666666,
        "bop19_average_recall": 0.6527722222222222,
    },
    "lmo": {
        "bop19_average_recall_vsd": 0.3976885813148789,
        "bop19_average_recall_mssd": 0.49411764705882344,
        "bop19_average_recall_mspd": 0.6067128027681661,
        "bop19_average_recall": 0.49950634371395614,
    },
}

# Loop through each entry in the dictionary and execute the command
for dataset_name, file_name in tqdm(FILE_DICTIONARY.items(), desc="Executing..."):
    output_file_name = f"{OUTPUT_DIR}/eval_bop19_pose_test_{dataset_name}.txt"
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

    with open(output_file_name, "a") as output_file:
        subprocess.run(command, stdout=output_file, stderr=subprocess.STDOUT)
print("Script executed successfully.")


# Check scores for each dataset
for dataset_name, _ in tqdm(FILE_DICTIONARY.items(), desc="Verifying..."):
    log_file_path = f"{OUTPUT_DIR}/eval_bop19_pose_test_{dataset_name}.txt"

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
