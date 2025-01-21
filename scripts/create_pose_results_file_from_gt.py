# Author: Mederic Fourmy (mederic.fourmy@gmail.com)
# Czech Technical University in Prague

"""
Create POSE result files from ground truth annotation and targets file.
"""

import os
import argparse

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout


# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
    # Out perfect result file name 
    "results_name": 'gt-results',    
    # Predefined test targets 
    "targets_filename": "test_targets_bop19.json",    
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    "dataset": "ycbv",
    "split": "test",  
    "split_type": None,
    # by default, we consider only objects that are at least 10% visible
    "visib_gt_min": 0.1,
    "eval_mode": "localization",
}
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", default=p["results_path"])
parser.add_argument("--results_name", default=p["results_name"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
parser.add_argument("--dataset", default=p["dataset"])
parser.add_argument("--split", default=p["split"])
parser.add_argument("--split_type", default=p["split_type"])
parser.add_argument("--visib_gt_min", default=p["visib_gt_min"])
parser.add_argument("--eval_mode", default=p["eval_mode"])
args = parser.parse_args()


p["results_path"] = str(args.results_path)
p["results_name"] = str(args.results_name)
p["targets_filename"] = str(args.targets_filename)
p["dataset"] = str(args.dataset)
p["split"] = str(args.split)
p["split_type"] = str(args.split_type) if args.split_type is not None else None
p["visib_gt_min"] = float(args.visib_gt_min)
p["eval_mode"] = str(args.eval_mode)

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p["datasets_path"], p["dataset"], p["split"], p["split_type"]
)

targets_path = os.path.join(p["datasets_path"], p["dataset"], p["targets_filename"])
targets = inout.load_json(targets_path)

unique_scene_ids = set([t["scene_id"] for t in targets])

scene_gts = {}
scene_gts_info = {}
results = []

for target in targets:
    scene_id, im_id = target["scene_id"], target["im_id"] 

    tpath_keys = dataset_params.scene_tpaths_keys(dp_split["eval_modality"], scene_id)

    if scene_id not in scene_gts:
        scene_gts[scene_id] = inout.load_scene_gt(
            dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id)
        )
        scene_gts_info[scene_id] = inout.load_scene_gt(
            dp_split[tpath_keys["scene_gt_info_tpath"]].format(scene_id=scene_id)
        )

    img_gt = scene_gts[scene_id][im_id]
    img_gt_info = scene_gts_info[scene_id][im_id]
    
    if "obj_id" not in target:
        target = inout.get_im_targets(img_gt, img_gt_info, p["visib_gt_min"], p["eval_mode"])

    for obj_gt in img_gt:
        if obj_gt["obj_id"] == target["obj_id"]:
            result = {
                "scene_id": int(scene_id),
                "im_id": int(im_id),
                "obj_id": int(obj_gt["obj_id"]),
                "score": 1.0,
                "R": obj_gt["cam_R_m2c"],
                "t": obj_gt["cam_t_m2c"],
                "time": -1.0,
            }
            results.append(result)

result_filename = "{}_{}-{}_pose.csv".format(p["results_name"], p["dataset"], p["split"])
results_path = os.path.join(p["results_path"], result_filename)
inout.save_bop_results(results_path, results)
print('Saved ', results_path)
