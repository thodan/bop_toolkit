#!/bin/sh
python scripts/create_pose_results_file_from_gt.py --dataset lmo --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset ycbv --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset tless --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset itodd --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset hb --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset icbin --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset tudl --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt

python scripts/create_coco_results_file_from_gt.py --dataset lmo --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_coco_results_file_from_gt.py --dataset ycbv --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_coco_results_file_from_gt.py --dataset tless --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_coco_results_file_from_gt.py --dataset itodd --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_coco_results_file_from_gt.py --dataset hb --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_coco_results_file_from_gt.py --dataset icbin --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_coco_results_file_from_gt.py --dataset tudl --results_path bop_toolkit_lib/tests/data/results_gt