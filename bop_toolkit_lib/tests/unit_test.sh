#!/bin/bash

# Define the input directory
INPUT_DIR="./bop_toolkit_lib/tests/data/"

# Define the output text file
OUTPUT_FILE="./bop_toolkit_lib/tests/unit_test_output.txt"

# List of file names
FILE_NAMES=(
    "cnos-fastsammegapose_icbin-test_7c9f443f-b900-41bb-af01-09b8eddfc2c4.csv"
    "cnos-fastsammegapose_tudl-test_1328490c-bf88-46ce-a12c-a5e5a7712220.csv"
    "cnos-fastsammegapose_lmo-test_16ab01bd-f020-4194-9750-d42fc7f875d2.csv"
    "cnos-fastsammegapose_ycbv-test_8fe0af14-16e3-431a-83e7-df00e93828a6.csv"
    "cnos-fastsammegapose_tless-test_94e046a0-42af-495f-8a35-11ce8ee6f217.csv"
)

# Loop through each file name and execute the command
for FILE_NAME in "${FILE_NAMES[@]}"; do
    python scripts/eval_bop19_pose.py --renderer_type=vispy --results_path $INPUT_DIR --eval_path $INPUT_DIR --result_filenames=$FILE_NAME --num_worker 10 >> $OUTPUT_FILE 2>&1
done