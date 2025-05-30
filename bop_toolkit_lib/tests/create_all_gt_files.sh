#!/bin/sh

# POSE
## Classic
python scripts/create_pose_results_file_from_gt.py --dataset lmo --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset ycbv --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset tless --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset itodd --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset hb --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset icbin --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset tudl --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt
## H3
python scripts/create_pose_results_file_from_gt.py --dataset hopev2 --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset hot3d --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset handal --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt
## Industrial
python scripts/create_pose_results_file_from_gt.py --dataset ipd --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset xyzibd --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt
python scripts/create_pose_results_file_from_gt.py --dataset itoddmv --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt


# COCO
## Classic
python scripts/create_coco_results_file_from_gt.py --dataset lmo --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt --compress
python scripts/create_coco_results_file_from_gt.py --dataset ycbv --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt --compress
python scripts/create_coco_results_file_from_gt.py --dataset tless --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt --compress
python scripts/create_coco_results_file_from_gt.py --dataset itodd --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt --compress
python scripts/create_coco_results_file_from_gt.py --dataset hb --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt --compress
python scripts/create_coco_results_file_from_gt.py --dataset icbin --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt --compress
python scripts/create_coco_results_file_from_gt.py --dataset tudl --targets_filename test_targets_bop19.json --results_path bop_toolkit_lib/tests/data/results_gt --compress
## H3
python scripts/create_coco_results_file_from_gt.py --dataset hopev2 --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt --ann_type bbox
python scripts/create_coco_results_file_from_gt.py --dataset hot3d --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt --ann_type bbox
python scripts/create_coco_results_file_from_gt.py --dataset handal --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt --ann_type bbox
## Industrial
python scripts/create_coco_results_file_from_gt.py --dataset ipd --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt --ann_type bbox
python scripts/create_coco_results_file_from_gt.py --dataset xyzibd --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt --ann_type bbox
python scripts/create_coco_results_file_from_gt.py --dataset itoddmv --targets_filename test_targets_bop24.json --results_path bop_toolkit_lib/tests/data/results_gt --ann_type bbox
