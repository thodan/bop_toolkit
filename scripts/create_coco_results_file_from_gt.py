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
args = parser.parse_args()

p["results_path"] = str(args.results_path)
p["results_name"] = str(args.results_name)
p["targets_filename"] = str(args.targets_filename)
p["dataset"] = str(args.dataset)
p["split"] = str(args.split)
p["split_type"] = str(args.split_type) if args.split_type is not None else None
p["bbox_type"] = str(args.bbox_type)

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], p["dataset"], p["split"], p["split_type"]
)
if not os.path.exists(dp_split["base_path"]):
    misc.log(f'Dataset does not exist: {dp_split["base_path"]}')
    exit()

# Load and organize the estimation targets.
target_file_path = os.path.join(dp_split["base_path"], p["targets_filename"])
targets = inout.load_json(target_file_path)
targets_org = {}
for target in targets:
    targets_org.setdefault(target["scene_id"], {}).setdefault(target["im_id"], {})
results = []

# loop over coco annotation and select based on targets
for scene_id in targets_org:
    tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], dp_split["eval_sensor"], scene_id)

    coco_gt_path = dp_split[tpath_keys["scene_gt_coco_tpath"]].format(scene_id=scene_id)
    if p["bbox_type"] == "modal":
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
                "segmentation": ann['segmentation'], 
                "time": -1, 
                "scene_id": scene_id
            }
            results.append(result)

if not os.path.exists(p["results_path"]):
    misc.log(f"Creating dir {p['results_path']}")
    os.mkdir(p["results_path"])
result_filename = f"{p['results_name']}_{p['dataset']}-{p['split']}_coco.json"
results_path = os.path.join(p["results_path"], result_filename)
inout.save_json(results_path, results)
result_file_path = os.path.join(p["results_path"], result_filename)
check_passed, _ = inout.check_coco_results(result_file_path, ann_type="segm")
if not check_passed:
    misc.log(f"Please correct the coco result format of {result_filename}")
    exit()
misc.log(f"Saved {results_path}")
