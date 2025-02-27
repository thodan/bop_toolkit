# Author: Mederic Fourmy (mederic.fourmy@gmail.com)
# Czech Technical University in Prague

"""
Create COCO format result files from ground truth annotation and targets file.
"""

import os
import argparse

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc


# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
    # Dataset name. See dataset_params.py for options.
    "dataset": "xyzibd",
    # Dataset split. See dataset_params.py for options
    "split": "test",  
    # Dataset split type. See dataset_params.py for options
    "split_type": None,
    # Out perfect result file name 
    "results_name": 'gt-results',    
    # Predefined test targets 
    "targets_filename": "test_targets_bop19.json",    
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # bbox type. Options: 'modal', 'amodal'.
    "bbox_type": "amodal",
    # Include segmentation masks in the result file.
    "ann_type": "segm",  # Options: 'bbox', 'segm'.
    # Save the result file in GZIP format.
    "compress": False,
}
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=p["dataset"])
parser.add_argument("--split", default=p["split"])
parser.add_argument("--split_type", default=p["split_type"])
parser.add_argument("--results_name", default=p["results_name"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
parser.add_argument("--results_path", default=p["results_path"])
parser.add_argument("--bbox_type", default=p["bbox_type"])
parser.add_argument("--ann_type", default=p["ann_type"])
parser.add_argument("--compress", action="store_true", default=p["compress"])
args = parser.parse_args()

split_type = str(args.split_type) if args.split_type is not None else None

misc.log(f"Creating coco {args.ann_type} results from gt for {args.dataset}")

assert args.ann_type in ["segm", "bbox"]



# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], args.dataset, args.split, split_type
)
if not os.path.exists(dp_split["base_path"]):
    misc.log(f'Dataset does not exist: {dp_split["base_path"]}')
    exit()

# Load and organize the estimation targets.
target_file_path = os.path.join(dp_split["base_path"], args.targets_filename)
targets = inout.load_json(target_file_path)
targets_org = misc.reorganize_targets(targets)

results = []

for scene_id in targets_org:
    tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)

    coco_gt_path = dp_split[tpath_keys["scene_gt_coco_tpath"]].format(scene_id=scene_id)
    if args.bbox_type == "modal":
        coco_gt_path = coco_gt_path.replace("scene_gt_coco", "scene_gt_coco_modal")
    scene_coco_ann = inout.load_json(coco_gt_path)['annotations']
    
    for ann in scene_coco_ann:
        # coco annotation -> list of dictionnary with keys 
        # ['id', 'image_id', 'category_id', 'iscrowd', 'area', 'bbox', 'segmentation', 'width', 'height', 'ignore'] 
        image_id = ann['image_id']
        if image_id in targets_org[scene_id]:
            result = {
                "image_id": image_id, 
                "bbox": ann['bbox'], 
                "score": 1.0, 
                "category_id": ann['category_id'], 
                "time": -1, 
                "scene_id": scene_id
            }
            if args.ann_type == 'segm':
                result["segmentation"] = ann['segmentation']
            results.append(result)

if not os.path.exists(args.results_path):
    misc.log(f"Creating dir {p['results_path']}")
    os.mkdir(args.results_path)
result_filename = f"{args.results_name}_{args.dataset}-{p['split']}_coco.json"
results_path = os.path.join(args.results_path, result_filename)
inout.save_json(results_path, results, args.compress, verbose=True)
result_file_path = os.path.join(args.results_path, result_filename)
check_passed, _ = inout.check_coco_results(result_file_path, ann_type=args.ann_type)
if not check_passed:
    misc.log(f"Please correct the coco result format of {result_filename}")
    exit()
